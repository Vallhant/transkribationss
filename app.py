from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import soundfile as sf
import numpy as np
from vosk import Model, KaldiRecognizer
import wave
import requests
import zipfile
import io
import shutil
import time
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Создаем директории, если их нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

def download_model():
    model_path = 'model'
    if not os.path.exists(os.path.join(model_path, 'conf')):
        print("Загрузка модели Vosk...")
        model_url = "https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip"
        response = requests.get(model_url, stream=True)
        
        # Создаем временную директорию для распаковки
        temp_dir = 'temp_model'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Сохраняем и распаковываем архив
        zip_path = os.path.join(temp_dir, 'model.zip')
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Перемещаем содержимое из временной директории в model
        extracted_dir = os.path.join(temp_dir, 'vosk-model-ru-0.22')
        for item in os.listdir(extracted_dir):
            s = os.path.join(extracted_dir, item)
            d = os.path.join(model_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        
        # Очищаем временные файлы
        shutil.rmtree(temp_dir)
        print("Модель успешно загружена!")
    return model_path

# Загружаем модель Vosk
model_path = download_model()
model = Model(model_path)

def combine_words_into_phrases(words):
    phrases = []
    current_phrase = []
    current_start = None
    current_end = None
    
    for word in words:
        if current_start is None:
            current_start = word['start']
            current_end = word['end']
            current_phrase.append(word['word'])
        else:
            # Если пауза между словами меньше 1 секунды, считаем их частью одной фразы
            if word['start'] - current_end < 1.0:
                current_phrase.append(word['word'])
                current_end = word['end']
            else:
                # Если пауза больше 1 секунды, начинаем новую фразу
                phrases.append({
                    'text': ' '.join(current_phrase),
                    'start': current_start,
                    'end': current_end
                })
                current_phrase = [word['word']]
                current_start = word['start']
                current_end = word['end']
    
    # Добавляем последнюю фразу
    if current_phrase:
        phrases.append({
            'text': ' '.join(current_phrase),
            'start': current_start,
            'end': current_end
        })
    
    return phrases

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if file and file.filename.endswith('.mp3'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Конвертируем MP3 в WAV с помощью ffmpeg напрямую
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{int(time.time())}.wav')
            os.system(f'ffmpeg -i "{filepath}" -acodec pcm_s16le -ar 16000 -ac 1 "{wav_path}"')
            
            # Транскрибация
            wf = wave.open(wav_path, "rb")
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            
            all_words = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result and 'result' in result:
                        all_words.extend(result['result'])
            
            final_result = json.loads(rec.FinalResult())
            if final_result and 'result' in final_result:
                all_words.extend(final_result['result'])
            
            # Закрываем файл перед удалением
            wf.close()
            
            # Даем системе время на освобождение файла
            time.sleep(0.1)
            
            # Очистка временных файлов
            try:
                os.remove(filepath)
                os.remove(wav_path)
            except Exception as e:
                print(f"Ошибка при удалении временных файлов: {e}")
            
            # Объединяем слова в фразы
            phrases = combine_words_into_phrases(all_words)
            
            return jsonify({'transcription': phrases})
            
        except Exception as e:
            return jsonify({'error': f'Ошибка при обработке файла: {str(e)}'}), 500
    
    return jsonify({'error': 'Неподдерживаемый формат файла'}), 400

if __name__ == '__main__':
    app.run(debug=True)
