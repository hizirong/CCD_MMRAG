import whisper

def transcribe_file(file_path, model_size="tiny"):
    model = whisper.load_model(model_size)
    result = model.transcribe(file_path)
    return result["text"]

def transcribe_file_nbest(file_path, model_size="tiny", num_beams=5):
    """使用n-best方式進行轉錄"""
    model = whisper.load_model(model_size)
    
    # 設置decode options來獲取多個結果
    options = dict(
        beam_size=num_beams,  # beam search 的寬度
        best_of=num_beams,    # 返回最好的n個結果
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8],  # 使用不同的temperature獲得更多樣的結果
        language="Chinese"     # 指定語言
    )
    
    # 進行轉錄
    result = model.transcribe(
        file_path,
        **options
    )
    
    # 收集所有結果
    transcriptions = []
    
    # 添加最佳結果
    transcriptions.append({
        'text': result['text'],
        'confidence': 1.0
    })
    
    # 添加其他候選結果（如果有的話）
    if hasattr(result, 'alternatives'):
        for i, alt in enumerate(result.alternatives):
            transcriptions.append({
                'text': alt.text,
                'confidence': 1.0 - ((i + 1) * 0.1)  # 簡單的置信度計算
            })
    
    return transcriptions
def transcribe_with_steps(file_path, model_size="tiny"):
    model = whisper.load_model(model_size)
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    
    # 計算 log mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # 檢測語言
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs.items(), key=lambda x: x[1])[0]}")
    
    # 解碼選項
    decode_options = {
        "language": "zh",
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.7,
        "without_timestamps": True
    }
    
    # 進行解碼並獲取所有候選
    transcribe_result = model.transcribe(
        file_path,
        **decode_options,
        fp16=False  # 避免 FP16 警告
    )
    
    # 打印解碼過程中的所有候選結果
    if hasattr(transcribe_result, "decode_candidates"):
        for i, candidate in enumerate(transcribe_result.decode_candidates):
            print(f"\nCandidate {i + 1}:")
            print(f"Text: {candidate.text}")
            print(f"Score: {candidate.score:.4f}")
    else:
        print("\nNo intermediate candidates available")
    
    return transcribe_result["text"]
def main():
    # 測試兩種方法
    # audio_file = "no_upload/test_mp3/01.mp3"
    
    # print("Original transcription:")
    # transcription = transcribe_file(audio_file)
    # print(transcription)
    
    # print("\nN-best transcriptions:")
    # nbest_results = transcribe_file_nbest(audio_file)
    # for i, result in enumerate(nbest_results):
    #     print(f"\nResult {i+1}:")
    #     print(f"Text: {result['text']}")
    #     print(f"Confidence: {result['confidence']:.2f}")
    audio_file = "no_upload/test_mp3/01.mp3"
    final_text = transcribe_with_steps(audio_file)
    print("\nFinal result:", final_text)

if __name__ == "__main__":
    main()