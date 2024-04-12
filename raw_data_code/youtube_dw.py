from yt_dlp import YoutubeDL
import sys 





def main(url):
    with YoutubeDL() as ydl:
        result = ydl.download([url])

if __name__ == '__main__':
    url = sys.argv[1]
    main(url)