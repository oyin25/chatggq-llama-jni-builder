#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <android/log.h>

#include "llama.h"

#define LOG_TAG "ChatGGQ-LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static std::atomic<bool> g_stop(false);
static std::mutex g_mutex;

extern "C"
JNIEXPORT jlong JNICALL
Java_com_lodwickmasete_llms_LlamaEngine_initModel(
        JNIEnv *env,
        jobject thiz,
        jstring modelPath
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) {
        return 0;
    }

    LOGI("Loading model: %s", path);

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(path, model_params);

    env->ReleaseStringUTFChars(modelPath, path);

    if (g_model == nullptr) {
        LOGE("Failed to load model");
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    ctx_params.no_perf = true;

    g_ctx = llama_init_from_model(g_model, ctx_params);

    if (g_ctx == nullptr) {
        LOGE("Failed to create llama context");
        llama_model_free(g_model);
        g_model = nullptr;
        return 0;
    }

    g_stop = false;

    return reinterpret_cast<jlong>(g_ctx);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_lodwickmasete_llms_LlamaEngine_stopGeneration(
        JNIEnv *env,
        jobject thiz
) {
    g_stop = true;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_lodwickmasete_llms_LlamaEngine_freeModel(
        JNIEnv *env,
        jobject thiz
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    g_stop = true;

    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

static bool contains_stop_text(const std::string &text, JNIEnv *env, jobjectArray stopStrings) {
    if (stopStrings == nullptr) return false;

    jsize count = env->GetArrayLength(stopStrings);

    for (jsize i = 0; i < count; i++) {
        jstring stop = (jstring) env->GetObjectArrayElement(stopStrings, i);
        if (stop == nullptr) continue;

        const char *stop_chars = env->GetStringUTFChars(stop, nullptr);

        if (stop_chars != nullptr) {
            std::string stop_text(stop_chars);
            env->ReleaseStringUTFChars(stop, stop_chars);

            if (!stop_text.empty() && text.find(stop_text) != std::string::npos) {
                env->DeleteLocalRef(stop);
                return true;
            }
        }

        env->DeleteLocalRef(stop);
    }

    return false;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_lodwickmasete_llms_LlamaEngine_runPromptStreaming(
        JNIEnv *env,
        jobject thiz,
        jstring prompt,
        jint maxTokens,
        jfloat temperature,
        jfloat topP,
        jint topK,
        jfloat repeatPenalty,
        jfloat freqPenalty,
        jfloat presencePenalty,
        jint penaltyLastN,
        jobjectArray stopStrings,
        jintArray extraStopTokenIds,
        jobject callback
) {
    if (g_ctx == nullptr || g_model == nullptr || callback == nullptr) {
        return;
    }

    const char *prompt_chars = env->GetStringUTFChars(prompt, nullptr);
    if (prompt_chars == nullptr) {
        return;
    }

    std::string input(prompt_chars);
    env->ReleaseStringUTFChars(prompt, prompt_chars);

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");

    if (onToken == nullptr) {
        LOGE("Callback onToken method not found");
        return;
    }

    g_stop = false;

    const llama_vocab *vocab = llama_model_get_vocab(g_model);

    int n_prompt = -llama_tokenize(
            vocab,
            input.c_str(),
            (int) input.size(),
            nullptr,
            0,
            true,
            true
    );

    if (n_prompt <= 0) {
        LOGE("Failed to count prompt tokens");
        return;
    }

    std::vector<llama_token> prompt_tokens(n_prompt);

    int tokenized = llama_tokenize(
            vocab,
            input.c_str(),
            (int) input.size(),
            prompt_tokens.data(),
            (int) prompt_tokens.size(),
            true,
            true
    );

    if (tokenized < 0) {
        LOGE("Failed to tokenize prompt");
        return;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;

    llama_sampler *sampler = llama_sampler_chain_init(sparams);

    if (temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(topK));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), tokenized);

    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("llama_decode failed for prompt");
        llama_sampler_free(sampler);
        return;
    }

    std::string generated;

    for (int i = 0; i < maxTokens; i++) {
        if (g_stop) {
            break;
        }

        llama_token new_token_id = llama_sampler_sample(sampler, g_ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[512];

        int n = llama_token_to_piece(
                vocab,
                new_token_id,
                buf,
                sizeof(buf),
                0,
                true
        );

        if (n > 0) {
            std::string piece(buf, n);
            generated += piece;

            if (contains_stop_text(generated, env, stopStrings)) {
                break;
            }

            jstring tokenStr = env->NewStringUTF(piece.c_str());
            env->CallVoidMethod(callback, onToken, tokenStr);
            env->DeleteLocalRef(tokenStr);
        }

        llama_batch next_batch = llama_batch_get_one(&new_token_id, 1);

        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("llama_decode failed during generation");
            break;
        }
    }

    llama_sampler_free(sampler);
}
