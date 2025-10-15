## Debate_scraper/
### + debate_scraper/metadata.py

This class helps us manage our metadata file within ```the debate_dataset``` folder. The metadata entries look like this:
```json
[
  {
    "video_id": "smkyorC5qwc",
    "title": "The Third Presidential Debate: Hillary Clinton And Donald Trump (Full Debate) | NBC News",
    "duration": 6958,
    "upload_date": "20161020",
    "channel": "NBC News",
    "url": "https://www.youtube.com/watch?v=smkyorC5qwc",
    "audio_path": "debate_dataset/audio/smkyorC5qwc.wav",
    "transcript_path": "debate_dataset/transcripts/smkyorC5qwc.json",
    "transcript_segments": 3932
  }, (... more entries) ]
```

### + debate_scraper/downloader.py

We also need a downloader class to download the audio files. Passing some options to the YoutubeDL downloader to ensure audio quality and finally doing our first scraping for the audio files:

```python
class AudioDownloader:
  (...)
    ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self.audio_format,
                    "preferredquality": self.quality,
                }],
                "outtmpl": str(self.output_dir / f"{video_id}.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                print(f"  ✓ Audio downloaded")
                return str(output_path)
            except Exception as e:
                print(f"Error downloading audio: {e}")
                return None
```

### + debate_scraper/transcript_fetcher.py

Just as the downloader class we need another class for the download of the transcripts: 

```python
class TranscriptFetcher:
  (...)
    try:
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.list(video_id)
                
                # Get transcript based on preference
                if self.manual_only:
                    transcript = transcript_list.find_manually_created_transcript(['en'])
                else:
                    transcript = transcript_list.find_transcript(['en'])
                
                return transcript.fetch()
```

### + debate_scraper/scraper.py

The ```DebateScraper``` class combines everything together and adds methods to handle youtube playlist and channels urls:

```python
def process_playlist(self, playlist_url, max_videos=None):
        """Process all videos in a playlist"""        
        videos = self._get_playlist_videos(playlist_url, max_videos)       
        if not videos:
            return            
        success_count = 0
        for video in tqdm(videos, desc="Processing videos"):
            if video:
                video_url = f"https://www.youtube.com/watch?v={video['id']}"
                if self.process_video(video_url):
                    success_count += 1
                time.sleep(2)
    
    def process_channel(self, channel_url, max_videos=None):
        """Process videos from a channel"""
        videos = self._get_channel_videos(channel_url, max_videos)       
        if not videos:
            return                
        success_count = 0
        for video in tqdm(videos, desc="Processing videos"):
            if video:
                video_url = f"https://www.youtube.com/watch?v={video['id']}"
                if self.process_video(video_url):
                    success_count += 1
                time.sleep(2)
```
The scraper in action:

```ts
Processing Presidential Debates...

Processing: smkyorC5qwc
  Title: The Third Presidential Debate: Hillary Clinton And Donald Trump (Full Debate) | NBC News
  Duration: 115 minutes
Fetching transcript...
Transcript saved (3932 segments)
Downloading audio...
  ✓ Audio downloaded
Successfully processed!

Processing: wW1lY5jFNcQ
  Title: First 2020 Presidential Debate between Donald Trump and Joe Biden
  Duration: 124 minutes
Fetching transcript...
Transcript saved (3126 segments)
Downloading audio...
  ✓ Audio downloaded
Successfully processed!

Processing: MAm5OCgZ6so
  Title: ABC Presidential Debate Simulcast
  Duration: 165 minutes
Fetching transcript...
No manual English transcript found

==================================================
DATASET STATISTICS
==================================================
Total videos: 2
Total duration: 4.0 hours
Total transcript segments: 7,058
Average segments per video: 3529
==================================================

(...)
```

## Dataset Folder
My folder structure for my raw data now looks like this.
```ts
debate_dataset/
├── audio/
    ├── first.wav
    └── second.wav
├── transcripts/
    ├── first_trnscrpt.json
    └── second_trnscrpt.json
└── metadata.json
```













            
