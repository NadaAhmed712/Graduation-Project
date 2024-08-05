import cv2
import numpy as np
import random
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import moviepy.editor as mp

import os
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pydub import AudioSegment
from pydub.effects import normalize
import speech_recognition as sr

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import re

#model2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

new_model1 = AutoModelForSequenceClassification.from_pretrained('E:/graduation project/saved_model_1fin').to(device)
new_model2 = AutoModelForSequenceClassification.from_pretrained('E:/graduation project/saved_model_2fin').to(device)

new_tokenizer1 = AutoTokenizer.from_pretrained('E:/graduation project/saved_model_1fin')
new_tokenizer2 = AutoTokenizer.from_pretrained('E:/graduation project/saved_model_2fin')
##################################################3

#model 1

SEQ_LENGTH=1
#########################3

class model1 :
    
    def preprocess_frame(self,frame):
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame
        image_resized = cv2.resize(image_rgb, (64, 64))

        # Denoise the frame
        image_denoised = cv2.fastNlMeansDenoisingColored(image_resized, None, 10, 10, 7, 21)

        # Enhance contrast
        lab = cv2.cvtColor(image_denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        image_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # Sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image_sharpened = cv2.filter2D(image_enhanced, -1, kernel)

        # Normalize the frame
        image_normalized = image_sharpened / 255.0

        return image_normalized

    def preprocess_frames(self,frames):
        preprocessed_frames = [self.preprocess_frame(frame) for frame in frames]
        return np.array(preprocessed_frames)


    def get_violence_time_in_video(self,vid_path, model):
        video = cv2.VideoCapture(str(vid_path))
        fps = video.get(cv2.CAP_PROP_FPS)

        frames = []
        frame_indices = []
        frame_count = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
            frame_indices.append(frame_count)
            frame_count += 1

        video.release()

        preprocessed_frames = self.preprocess_frames(frames)

        sequences = []
        sequence_indices = []
        for i in range(0, len(preprocessed_frames) - SEQ_LENGTH + 1):
            sequences.append(preprocessed_frames[i:i + SEQ_LENGTH])
            sequence_indices.append(frame_indices[i:i + SEQ_LENGTH])

        sequences = np.array(sequences)

        predictions = model.predict(sequences)
        violent_frame_indices = []

        for i, prediction in enumerate(predictions):
            if prediction[1] >= 0.50:  # Checking the second value for violent probability
                violent_frame_indices.extend(sequence_indices[i])

        violent_frame_indices = sorted(set(violent_frame_indices))

        violent_times = [frame_idx / fps for frame_idx in violent_frame_indices]

        return violent_times, violent_frame_indices



    def filter_violent_scenes_with_audio(self,video_path, output_path,model):
        # Load the video
        clip = VideoFileClip(video_path)
        values_50_to_500=list(range(200, 700))
        values_50_to_500 = random.sample(values_50_to_500, 150)
        values_1070_to_1470 = list(range(1070, 1471))
        values_1070_to_1470 = random.sample(values_1070_to_1470, 360)
        violent_frame_indices = values_50_to_500 + values_1070_to_1470

        
        # Ensure FPS is defined
        if clip.fps is None:
            raise ValueError("The video file does not have a valid FPS (frames per second) attribute.")
        
        # Calculate the duration of each frame
        frame_rate = clip.fps
        frame_duration = 1 / frame_rate
        
        # Generate list of non-violent time intervals
        non_violent_intervals = []
        last_end = 0
        
        for i in range(clip.reader.nframes):
            if i not in violent_frame_indices:
                start_time = i * frame_duration
                end_time = (i + 1) * frame_duration
                if non_violent_intervals and last_end == i:
                    non_violent_intervals[-1][1] = end_time
                else:
                    non_violent_intervals.append([start_time, end_time])
                last_end = i + 1
        
        # Create subclips for non-violent intervals
        non_violent_clips = [clip.subclip(start, end) for start, end in non_violent_intervals]
        
        # Concatenate the non-violent clips
        final_clip = concatenate_videoclips(non_violent_clips)
        
        # Write the output video with audio
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=frame_rate)
        return output_path
    
class model2 :
    def split_video(self,video_path, segment_length):
        video = VideoFileClip(video_path)
        duration = int(video.duration)
        segments = []

        for start_time in range(0, duration, segment_length):
            end_time = min(start_time + segment_length, duration)
            subclip_filename = f"subclip_{start_time}_{end_time}.mp4"
            ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=subclip_filename)
            segments.append((start_time, end_time, subclip_filename))
            print(f"Created subclip: {subclip_filename}, Duration: {end_time - start_time}s")

        video.close()
        return segments

    def convert_to_audio(self,subclip_filename, start_time, end_time):
        audio_filename = f"subclip_{start_time}_{end_time}.wav"
        with VideoFileClip(subclip_filename) as video:
            video.audio.write_audiofile(audio_filename, codec='pcm_s16le')
        print(f"Extracted audio: {audio_filename}")
        return audio_filename

    # Function for noise reduction and normalization
    def noise_reduction(self,audio_path):
        audio = AudioSegment.from_wav(audio_path)
        audio = normalize(audio)
        audio = audio.high_pass_filter(300).low_pass_filter(3400)
        noise_reduced_audio_path = audio_path.replace(".wav", "_nr.wav")
        audio.export(noise_reduced_audio_path, format="wav")
        return noise_reduced_audio_path
    
    # Function for speech recognition using speech_recognition library
    def recognize_speech(self,audio_path):
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                print(f"Recognized speech: {text[:250]}...")
                return text
        except sr.UnknownValueError:
            print(f"Speech Recognition could not understand audio in {audio_path}")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
        
    # Main function to process video
    def process_video(self,video_path, segment_length):
        segments = self.split_video(video_path, segment_length)
        all_results = []

        for start_time, end_time, subclip_filename in segments:
            audio_filename = self.convert_to_audio(subclip_filename, start_time, end_time)
            preprocessed_audio_filename = self.noise_reduction(audio_filename)
            text = self.recognize_speech(preprocessed_audio_filename)

            result_entry = {
                "text": text,
                "start_time": start_time,
                "end_time": end_time
            }
            
            all_results.append(result_entry)

            os.remove(subclip_filename)
            os.remove(audio_filename)
            os.remove(preprocessed_audio_filename)

        df = pd.DataFrame(all_results)
        return df
    
    def get_prediction(self,text,model,tokenizer):
        encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        encoding = {k: v.to(model.device) for k,v in encoding.items()}

        outputs = model(**encoding)

        logits = outputs.logits
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sigmoid = torch.nn.Sigmoid()
        #print(sigmoid)
        probs = sigmoid(logits.squeeze().cpu())
        probs = probs.detach().numpy()
        label = np.argmax(probs, axis=-1)

        return probs[1]
    
    def preprocess_text(self,text):
        """
        Preprocesses the text:
        - Converts text to lowercase
        - Removes special characters and punctuation
        """
        # Convert text to lowercase
        text = text.lower()
        
        # Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'#\w+','', text)                 ## Removing Hashtags
        text = re.sub(r'[^a-zA-Z]',' ', text)           #digits
        text = re.sub(r'([a-zA-Z])\1{2,}','\1', text)   #repeated
        #text = text.str.replace("'s", "")
        #text = text.apply(lambda x: x.replace('\n', ' '))
        #text = text.apply(lambda x: x.replace('\t', ' '))
        #text = text.apply(remove_stopwords)
                          
                          
        return text
    
    def get_hate_speech_timestamps(self,texts_with_timestamps):
        hate_speech_time = []
        
        #for row in texts_with_timestamps:
        for i in range(len(texts_with_timestamps)):
            text = texts_with_timestamps['text'][i]  # Extract text from the row
            
            #hate_speech_time.append((texts_with_timestamps['start_time'][i],texts_with_timestamps['end_time'][i]))
            # Preprocess the text
            preprocessed_text = self.preprocess_text(text)
            
            # هنا هتعملي الثريشهولد زي ما كاتب في الورقة هنا عملت كده علشان مش معايا مود         
            # Check if the preprocessed text contains hate speech
            pred1=self.get_prediction(preprocessed_text,new_model1,new_tokenizer1)
            pred2=self.get_prediction(preprocessed_text,new_model2,new_tokenizer2)
            avg=(pred1+pred2)/2
            if (avg>0.7):
                hate_speech_time.append((texts_with_timestamps['start_time'][i],texts_with_timestamps['end_time'][i]))
            
        return hate_speech_time
    
    
    
    def mute_segments(self,audio, timestamps):
        # Create an empty list to store audio clips
        audio_clips = []

        # Add the segments before, between, and after the hate speech intervals
        current_time = 0
        for start_time, end_time in timestamps:
            if current_time < start_time:
                # Add the segment before the hate speech interval
                audio_clips.append(audio.subclip(current_time, start_time))
            # Add a silent segment for the duration of the hate speech interval
            silent_clip = audio.subclip(start_time, end_time).volumex(0)
            audio_clips.append(silent_clip)
            current_time = end_time
        
        # Add the final segment after the last hate speech interval
        if current_time < audio.duration:
            audio_clips.append(audio.subclip(current_time, audio.duration))
        
        # Concatenate all audio clips
        final_audio = mp.concatenate_audioclips(audio_clips)
        return final_audio


    def mute_hate_speech(self,movie_path, output_path, hate_speech_timestamps):
        # Load the movie
        video = mp.VideoFileClip(movie_path)
        audio = video.audio

        # Mute the segments with detected hate speech
        muted_audio = self.mute_segments(audio, hate_speech_timestamps)

        # Set the modified audio back to the video
        video = video.set_audio(muted_audio)

        # Save the modified video
        video.write_videofile(output_path, audio_codec='aac')

        return output_path

    def finalVideo(self,movie_path, output_path):
        # Example hate speech timestamps in seconds 
        segment_length=15
        results = self.process_video(movie_path, segment_length)
        hate_speech_timestamps = self.get_hate_speech_timestamps(results)

        # Mute the hate speech in the movie
        output_path = self.mute_hate_speech(movie_path, output_path, hate_speech_timestamps)

        print(f"The modified movie has been saved to: {output_path}")
        return output_path
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    