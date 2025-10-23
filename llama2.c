#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#ifdef RISCV_ROCKET
#include <riscv-pk/encoding.h>
#endif

#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>

#include "nn.h"
#include "profile.h"
#include "mico_nn.h"
#include "mico_quant.h"

#include "llama2_config.h"

#ifdef QUANTIZED
typedef Tensor2D_Q8 DataType;
typedef qbyte WeightType;
#else
typedef Tensor2D_F32 DataType;
typedef float WeightType;
#endif

#ifdef USE_INT8_KV
typedef int8_t kv_type;
#else
typedef float kv_type;
#endif

#ifndef LLAMA2_BIN
#define LLAMA2_BIN "./llama2/llama_model.bin"
#endif

INCLUDE_FILE(".rodata", LLAMA2_BIN, llama_model);
extern uint8_t llama_model_data[];
extern size_t llama_model_start[];
extern size_t llama_model_end[];

INCLUDE_FILE(".rodata", "./llama2/tokenizer.bin", tokenizer);
extern uint8_t tokenizer_data[];
extern size_t tokenizer_start[];
extern size_t tokenizer_end[];
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    DataType* wq; // (layer, dim, n_heads * head_size)
    DataType* wk; // (layer, dim, n_kv_heads * head_size)
    DataType* wv; // (layer, dim, n_kv_heads * head_size)
    DataType* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    DataType* w1; // (layer, hidden_dim, dim)
    DataType* w2; // (layer, dim, hidden_dim)
    DataType* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    DataType wcls;
} TransformerWeights;

typedef struct{
    qtype* wq_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* wk_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* wv_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* wo_qtype; // (layer * 2, i - wq, i + 1 - aq)
    // weights for ffn
    qtype* w1_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* w2_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* w3_qtype; // (layer * 2, i - wq, i + 1 - aq)

} TransformerQScheme;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    kv_type* key_cache;   // (layer, seq_len, dim)
    kv_type* value_cache; // (layer, seq_len, dim)
    #ifdef USE_INT8_KV
    float* key_scales; // (layer, seq_len)
    float* value_scales; // (layer, seq_len)
    #endif
    // RoPE caches
    float* rope_inv_freq; // (head_size/2)
    float* rope_cos; // (seq_len, head_size/2)
    float* rope_sin; // (seq_len, head_size/2)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    TransformerQScheme qscheme; // quantization scheme for the weights
    RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    return MiCo_time();
}

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = p->dim / p->n_heads;
    int head_pairs = head_size / 2;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    #ifdef USE_INT8_KV
    // Allocate Float Buffer for K V
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->key_scales = calloc(p->n_layers * p->seq_len, sizeof(float));
    s->value_scales = calloc(p->n_layers * p->seq_len, sizeof(float));
    printf("Alloacte KV Quant Buffer + Scales of size %ld Bytes...\n",
        (kv_dim * 2 + p->n_layers * p->seq_len * 2) * sizeof(float));
    #endif
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(kv_type));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(kv_type));
    
    printf("Alloacte KV Cache of size %ld KB...\n",
        (p->n_layers * p->seq_len * kv_dim * sizeof(kv_type)) / 1024);
    
    
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));

    // RoPE precompute: inv_freq per head (shared across heads)
    s->rope_inv_freq = calloc(head_pairs, sizeof(float));
    // Precompute cos/sin for all positions and per-head pairs
    size_t tbl_elems = (size_t)p->seq_len * head_pairs;
    s->rope_cos = calloc(tbl_elems, sizeof(float));
    s->rope_sin = calloc(tbl_elems, sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits
     || !s->rope_inv_freq || !s->rope_cos || !s->rope_sin) {
        printf("malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Precomputing RoPE tables (%ld Bytes)...\n", 
        tbl_elems * sizeof(float) * 2);
    long start = MiCo_time();
    // Fill inv_freq: 10000^(-2k/d_head), k in [0, head_size/2)
    for (int k = 0; k < head_pairs; ++k) {
        s->rope_inv_freq[k] = powf(10000.0f, -2.0f * (float)k / (float)head_size);
    }
    // Fill tables
    for (int pos = 0; pos < p->seq_len; ++pos) {
        for (int k = 0; k < head_pairs; ++k) {
            float angle = pos * s->rope_inv_freq[k];
            size_t idx = (size_t)pos * head_pairs + k;
            s->rope_cos[idx] = cosf(angle);
            s->rope_sin[idx] = sinf(angle);
        }
    }
    long end = MiCo_time();
    long total_run_state_size = p->dim * sizeof(float) * 4;
    total_run_state_size += p->hidden_dim * sizeof(float) * 2;
    total_run_state_size += p->n_layers * p->seq_len * kv_dim * sizeof(kv_type) * 2;
    total_run_state_size += p->n_heads * p->seq_len * sizeof(float);
    total_run_state_size += p->vocab_size * sizeof(float);
    total_run_state_size += head_pairs * sizeof(float) * (1 + 2 * p->seq_len);
    printf("Total Run State Size: %ld KB\n", total_run_state_size / 1024);
    printf("Done in %ld time\n", end - start);
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

size_t init_weight(DataType* w, char* ptr, int n_layers, int n, int m){
    char* ptr0 = ptr;
    for (int i = 0; i < n_layers; i++) {
        w[i].shape[0] = n;
        w[i].shape[1] = m;
        w[i].data = (WeightType*) ptr;
        ptr += n * m * sizeof(WeightType);
    }
    return ptr - ptr0;
}

size_t init_quant_weight(DataType* w, char* ptr, int n_layers, int n, int m){
    char* ptr0 = ptr;
    for (int i = 0; i < n_layers; i++) {
        w[i].shape[0] = n;
        w[i].shape[1] = m;
        // w[i].scale = *(float*)ptr;
        // ptr += sizeof(float);
        w[i].data = (WeightType*) ptr;
        ptr += n * m * sizeof(WeightType) / (8 / w[i].wq);
    }
    return ptr - ptr0;
}

size_t init_quant_weight_scale(DataType* w, char* ptr, int n_layers){
    char* ptr0 = ptr;
    for (int i = 0; i < n_layers; i++) {
        w[i].scale = *(float*)ptr;
        ptr += sizeof(float);
    }
    return ptr - ptr0;
}

void* init_float_params(
    TransformerWeights *w, 
    Config* p, 
    char* ptr, int shared_weights){

    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;

    w->rms_att_weight = (float*) ptr;
    ptr += n_layers * p->dim * sizeof(float);

    w->rms_ffn_weight = (float*) ptr;
    ptr += n_layers * p->dim * sizeof(float);

    w->rms_final_weight = (float*) ptr;
    ptr += p->dim * sizeof(float);

    return ptr;
}

void* init_qschemes(
    TransformerQScheme *wq,
    Config* p,
    char* ptr
){
    int n_layers = p->n_layers;

    wq->wq_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->wk_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->wv_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->wo_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->w1_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->w2_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->w3_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    if ((size_t)ptr % 4 != 0) {
        ptr += 4 - ((size_t)ptr % 4);
    }
    return ptr;
}

void init_weight_qtypes(qtype* wq, DataType*w, int n_layers){
    for (int i = 0; i < n_layers; i++) {
        w[i].wq = wq[2*i];
    }
}

void memory_map_weights(
    TransformerWeights *w, 
    TransformerQScheme *q,
    Config* p, 
    char* ptr, int shared_weights){
    
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;

    size_t inc = 0;
    w->wq = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->wq_qtype, w->wq, n_layers);
    ptr += init_quant_weight(w->wq, ptr, n_layers, p->dim, p->n_heads * head_size);
    #else
    ptr += init_weight(w->wq, ptr, n_layers, p->dim, p->n_heads * head_size);
    #endif

    w->wk = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->wk_qtype, w->wk, n_layers);
    ptr += init_quant_weight(w->wk, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    #else
    ptr += init_weight(w->wk, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    #endif

    w->wv = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->wv_qtype, w->wv, n_layers);
    ptr += init_quant_weight(w->wv, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    #else
    ptr += init_weight(w->wv, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    #endif

    w->wo = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->wo_qtype, w->wo, n_layers);
    ptr += init_quant_weight(w->wo, ptr, n_layers, p->n_heads * head_size, p->dim);
    #else
    ptr += init_weight(w->wo, ptr, n_layers, p->n_heads * head_size, p->dim);
    #endif

    w->w1 = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->w1_qtype, w->w1, n_layers);
    ptr += init_quant_weight(w->w1, ptr, n_layers, p->hidden_dim, p->dim);
    #else
    ptr += init_weight(w->w1, ptr, n_layers, p->hidden_dim, p->dim);
    #endif

    w->w2 = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->w2_qtype, w->w2, n_layers);
    ptr += init_quant_weight(w->w2, ptr, n_layers, p->dim, p->hidden_dim);
    #else
    ptr += init_weight(w->w2, ptr, n_layers, p->dim, p->hidden_dim);
    #endif

    w->w3 = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->w3_qtype, w->w3, n_layers);
    ptr += init_quant_weight(w->w3, ptr, n_layers, p->hidden_dim, p->dim);
    #else
    ptr += init_weight(w->w3, ptr, n_layers, p->hidden_dim, p->dim);
    #endif

    if(shared_weights){
        #ifdef QUANTIZED
        // Use INT8 for the final classifier to save memory
        w->wcls.shape[0] = p->vocab_size;
        w->wcls.shape[1] = p->dim;
        w->wcls.data = (WeightType*) ptr;
        w->wcls.wq = 8;
        ptr += p->vocab_size * p->dim * sizeof(WeightType);
        w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
        printf("Allocating Embedding Table of size %ld KB...\n",
            (p->vocab_size * p->dim * sizeof(float)) / 1024);
        #else
        w->token_embedding_table = (float*) ptr;
        w->wcls = w->token_embedding_table; // shared embedding table
        ptr += p->vocab_size * p->dim * sizeof(float);
        #endif
    }
    // Initialize Quantization Weight Scales
    #ifdef QUANTIZED
    ptr += init_quant_weight_scale(w->wq, ptr, n_layers);
    ptr += init_quant_weight_scale(w->wk, ptr, n_layers);
    ptr += init_quant_weight_scale(w->wv, ptr, n_layers);
    ptr += init_quant_weight_scale(w->wo, ptr, n_layers);
    ptr += init_quant_weight_scale(w->w1, ptr, n_layers);
    ptr += init_quant_weight_scale(w->w2, ptr, n_layers);
    ptr += init_quant_weight_scale(w->w3, ptr, n_layers);
    if(shared_weights){
        w->wcls.scale = *(float*)ptr;
        ptr += sizeof(float);
        for(int i = 0; i < p->vocab_size * p->dim; i++){
            int8_t v = ((int8_t*)w->wcls.data)[i];
            w->token_embedding_table[i] = v * w->wcls.scale;
        }
    }
    #endif

    return;
}

void read_checkpoint(Config* config, TransformerWeights* weights, TransformerQScheme* qsheme) {
    char* ptr = (char*)llama_model_start;
    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number = *(uint32_t*)ptr;
    ptr += sizeof(uint32_t);
    if (magic_number != 0x616b3432) { printf("Bad magic number\n"); exit(EXIT_FAILURE); }
    // read in the version number (uint32), has to be 2
    int version = *(int*)(ptr);
    if (version != 1) { printf( "Bad version %d, need version1\n", version); exit(EXIT_FAILURE); }
    ptr += sizeof(int);
    int header_size = 256; // the header size for version 2 in bytes
    // read in the config header
    if (memcpy(config, ptr, sizeof(Config)) == NULL) { exit(EXIT_FAILURE); }
    ptr += sizeof(Config);
    printf("Model Config:\ndim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d\n",
           config->dim, config->hidden_dim, config->n_layers, config->n_heads, config->n_kv_heads, config->vocab_size, config->seq_len);

    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // memory map the Transformer weights into the data pointer
    void* weights_ptr = (char*)llama_model_start + header_size; // skip header bytes. char is 1 byte
    weights_ptr = init_float_params(weights, config, weights_ptr, shared_weights);
    #ifdef QUANTIZED
    weights_ptr = init_qschemes(qsheme, config, weights_ptr);
    #endif
    memory_map_weights(weights, qsheme, config, weights_ptr, shared_weights);
}

// llama2 tokenizer
static int hex2int(char c){
    if(c >= '0' && c <= '9'){
        return c - '0';
    }else if(c >= 'a' && c <= 'f'){
        return c - 'a' + 10;
    }else if(c >= 'A' && c <= 'F'){
        return c - 'A' + 10;
    }
    return 0;
}

// llama2 tokenizer
int tokscanf(const char* piece, unsigned char* byte_val){
    // const char* pat = "<0x00>";
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>'){
        return 0;
    }
    char h = piece[3];
    char l = piece[4];
    // convert the hex char to int
    *byte_val = (hex2int(h) << 4) | hex2int(l);
    return 1;
}

void build_transformer(Transformer *t) {
    printf("Building Transformer model...\n");
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(&t->config, &t->weights, &t->qscheme);
    // allocate memory for the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // free the RunState buffers
    free_run_state(&t->state);
}
// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

// Some profilers
long RMSNORM_TIMER = 0;
long FMATMUL_TIMER = 0;
long ATTENTION_TIMER = 0;
long ROPE_TIMER = 0;

void init_timers() {
    SOFTMAX_TIMER = 0;
    QMATMUL_TIMER = 0;
    QUANT_TIMER = 0;
    RMSNORM_TIMER = 0;
    FMATMUL_TIMER = 0;
    SOFTMAX_TIMER = 0;
    ATTENTION_TIMER = 0;
    ROPE_TIMER = 0;
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    long start = MiCo_time();
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
    long end = MiCo_time();
    RMSNORM_TIMER += end - start;
}

void fmatmul(float* xout, float* x, Tensor2D_F32* w, int n, int d) {
    long start = MiCo_time();
    // Temporary Tensors
    Tensor2D_F32 Tx = { .shape = {1, n}, .data = x };
    Tensor1D_F32 Tb = { .shape = {0}, .data = NULL };
    Tensor2D_F32 Ty = { .shape = {1, d}, .data = xout };

    MiCo_linear_f32(&Ty, &Tx, w, &Tb);
    long end = MiCo_time();
    FMATMUL_TIMER += end - start;
}

void qmatmul(float* xout, float* x, Tensor2D_Q8* w, int n, int d, 
        qtype wq, qtype aq) {

    // Temporary Tensors
    Tensor2D_F32 Tx = { .shape = {1, n}, .data = x };
    Tensor1D_F32 Tb = { .shape = {0}, .data = NULL };
    Tensor2D_F32 Ty = { .shape = {1, d}, .data = xout };
    MiCo_bitlinear_f32(&Ty, &Tx, w, &Tb, wq, aq, 1); // TODO: alignment
}

float* forward(Transformer* transformer, int token, int pos) {

    init_timers();

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    TransformerQScheme* qscheme = &transformer->qscheme;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int head_pairs = head_size / 2;


    MiCo_MHA_Config mha_config = {
        .n_heads = p->n_heads,
        .head_size = head_size,
        .seq_len = p->seq_len,
        .kv_dim = kv_dim,
        .kv_mul = kv_mul
    };

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));
    long forward_start = MiCo_time();
    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        #ifdef RISCV_VEXII
        printf("Processing Layer %d/%d\n", (int)l+1, (int)p->n_layers);
        #endif
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        #ifdef USE_INT8_KV
        kv_type* qk_ptr = s->key_cache + loff + pos * kv_dim;
        kv_type* qv_ptr = s->value_cache + loff + pos * kv_dim;
        #else
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
        #endif
        // qkv matmuls for this position
        #ifdef QUANTIZED
        // TODO: The act quantization is redundant!
        qmatmul(s->q, s->xb, w->wq + l, dim, dim,
            qscheme->wq_qtype[2*l], qscheme->wq_qtype[2*l+1]);
        qmatmul(s->k, s->xb, w->wk + l, dim, kv_dim,
            qscheme->wk_qtype[2*l], qscheme->wk_qtype[2*l+1]);
        qmatmul(s->v, s->xb, w->wv + l, dim, kv_dim,
            qscheme->wv_qtype[2*l], qscheme->wv_qtype[2*l+1]);
        #else
        fmatmul(s->q, s->xb, w->wq + l, dim, dim);
        fmatmul(s->k, s->xb, w->wk + l, dim, kv_dim);
        fmatmul(s->v, s->xb, w->wv + l, dim, kv_dim);
        #endif
        
        #ifdef USE_INT8_KV
        // Quantize and store k and v into the kv cache
        long quant_start = MiCo_time();
        s->key_scales[l*p->seq_len + pos] = __FP32toQ8(qk_ptr, s->k, kv_dim);
        s->value_scales[l*p->seq_len + pos] = __FP32toQ8(qv_ptr, s->v, kv_dim);
        QUANT_TIMER += MiCo_time() - quant_start;
        #endif

        long rope_start = MiCo_time();
        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {

            int kpair = (i % head_size) >> 1; // pair index within the head
            size_t ridx = (size_t)pos * head_pairs + kpair;
            float fcr = s->rope_cos[ridx];
            float fci = s->rope_sin[ridx];

            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
        ROPE_TIMER += MiCo_time() - rope_start;
        // multihead attention. iterate over all heads
        long attn_start = MiCo_time();

        Tensor2D_F32 output = {
            .shape = {p->n_heads, head_size},
            .data = s->xb
        };
        Tensor2D_F32 query = {
            .shape = {p->n_heads, head_size},
            .data = s->q };

        #ifdef USE_INT8_KV
        MiCo_multihead_attention_f32_kv8(
            &output, 
            &query, 
            s->key_cache + loff, 
            s->value_cache + loff, 
            s->key_scales + l * p->seq_len,
            s->value_scales + l * p->seq_len,
            s->att,
            pos, 
            &mha_config
        );
        #else
        MiCo_multihead_attention_f32(
            &output, 
            &query, 
            s->key_cache + loff, 
            s->value_cache + loff, 
            s->att,
            pos, 
            &mha_config
        );
        #endif

        ATTENTION_TIMER += MiCo_time() - attn_start;

        // final matmul to get the output of the attention
        #ifdef QUANTIZED
        qmatmul(s->xb2, s->xb, w->wo + l, dim, dim,
            qscheme->wo_qtype[2*l], qscheme->wo_qtype[2*l+1]);
        #else
        fmatmul(s->xb2, s->xb, w->wo + l, dim, dim);
        #endif
        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm (ignored)
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);


        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        #ifdef QUANTIZED
        qmatmul(s->hb, s->xb, w->w1 + l, dim, hidden_dim,
            qscheme->w1_qtype[2*l], qscheme->w1_qtype[2*l+1]);
        qmatmul(s->hb2, s->xb, w->w3 + l, dim, hidden_dim,
            qscheme->w3_qtype[2*l], qscheme->w3_qtype[2*l+1]);
        #else
        fmatmul(s->hb, s->xb, w->w1 + l, dim, hidden_dim);
        fmatmul(s->hb2, s->xb, w->w3 + l, dim, hidden_dim);
        #endif
        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        // final matmul to get the output of the ffn
        #ifdef QUANTIZED
        qmatmul(s->xb, s->hb, w->w2 + l, hidden_dim, dim,
            qscheme->w2_qtype[2*l], qscheme->w2_qtype[2*l+1]);
        #else
        fmatmul(s->xb, s->hb, w->w2 + l, hidden_dim, dim);
        #endif
        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);
    // classifier into logits

    #ifdef RISCV_VEXII
    printf("Processing Final Classifier\n");
    #endif

    #ifdef QUANTIZED
    qmatmul(s->logits, x, &w->wcls, p->dim, p->vocab_size, w->wcls.wq, 8); // final classifier always uses 8-bit quant
    #else
    Tensor2D_F32 wcls = { .shape = {p->vocab_size, dim}, .data = w->wcls };
    fmatmul(s->logits, x, &wcls, p->dim, p->vocab_size);
    #endif
    long forward_end = MiCo_time();
    #ifdef RISCV_VEXII
    printf("Forward Time: %ld \n", (forward_end - forward_start));
    printf("QMatMul Time: %ld \n", QMATMUL_TIMER);
    printf("Quant Time: %ld \n", QUANT_TIMER);
    printf("Attention Time: %ld \n", ATTENTION_TIMER);
    printf("RMSNorm Time: %ld \n", RMSNORM_TIMER);
    printf("RoPE Time: %ld \n", ROPE_TIMER);
    printf("Softmax Time: %ld \n", SOFTMAX_TIMER);
    // printf("Final Float MatMul Time: %ld \n", FMATMUL_TIMER);
    #endif
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;
int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    unsigned char* ptr = (unsigned char*)tokenizer_start;
    // read in the file
    if (!ptr) {
         printf("couldn't load tokenizer\n"); exit(EXIT_FAILURE); 
    }
    if (memcpy(&t->max_token_length, ptr, sizeof(int)) == NULL) {
         printf("failed read\n"); exit(EXIT_FAILURE);
        }
    ptr += sizeof(int);
    // printf("max_token_length=%d\n", t->max_token_length);
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (memcpy(t->vocab_scores + i, ptr, sizeof(float)) == NULL) { printf("failed read\n"); exit(EXIT_FAILURE);}
        ptr += sizeof(float);
        if (memcpy(&len, ptr, sizeof(int) ) == NULL) { printf("failed read\n"); exit(EXIT_FAILURE); }
        ptr += sizeof(int);
        t->vocab[i] = (char *)malloc(len + 1);
        if (memcpy(t->vocab[i], ptr, len) == NULL) { printf("failed read\n"); exit(EXIT_FAILURE); }
        ptr += len;
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (tokscanf(piece, &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex* )bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { printf("cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}


void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int start_pos, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        printf("something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = start_pos;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        #ifdef RISCV_VEXII
        printf("Generated:");
        #endif
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        #ifdef RISCV_VEXII
        printf("\n");
        #endif
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        printf("achieved cycles per tok: %ld\n", (end-start)/(pos-1) );
    }

    free(prompt_tokens);
}

#ifdef USE_HOST
const int total_step = 128;
#else
const int total_step = 1;
#endif

int main(){
    printf("MiCo Transformer Demo\n");
    float temperature = 0.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 1.0f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int start_pos = 32;          // position in the sequence to start at, normally 0
    int steps = start_pos + total_step;     // number of steps to run for
    char *prompt = ""; // prompt string
    unsigned long long rng_seed = 42; // seed rng with time by default

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time_in_ms();
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length
    printf("Building Tokenizer and Sampler...\n");
    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    printf("Generating: \n");
    #ifdef REPEAT
    while(1){
    #endif
    generate(&transformer, &tokenizer, &sampler, prompt, start_pos, steps);
    #ifdef REPEAT
    }
    #endif
    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}