import kagglehub

def download_hm():
    path = kagglehub.competition_download(
        'h-and-m-personalized-fashion-recommendations'
    )
    print("Downloaded to:", path)

if __name__ == "__main__":
    download_hm()