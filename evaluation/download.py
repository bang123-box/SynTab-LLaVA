import os
import requests
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from concurrent.futures import ThreadPoolExecutor,  as_completed
from urllib.parse import urlparse

def download_image(name, url, folder):
    f = open(folder + ".txt", "a")
    try:
        i = 0
        while i < 1:
            # 发送GET请求下载图片
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # 构造图片保存路径
                image_path = os.path.join(folder, name + ".jpg")
                # 写入图片文件
                with open(image_path, 'wb') as fw:
                    for chunk in response.iter_content(1024):
                        fw.write(chunk)
                f.write(name + " " + "yes\n")
                break
            else:
                i += 1
        if i >= 1:
            f.write(name + " " + "no\n")
    except Exception as e:
        f.write(name + " " + "no\n")
    f.close()

def download_images(image_name, image_urls, folder, max_workers=20):
    # 创建保存图片的文件夹
    os.makedirs(folder, exist_ok=True)
    # 使用线程池管理下载任务
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交下载任务
        futures = [executor.submit(download_image, name, url, folder) for name, url in zip(image_name, image_urls)]
        
        # 创建进度条，追踪 futures 完成情况
        with tqdm(total=len(futures), desc='Downloading images') as pbar:
            # 遍历已完成的 futures
            for future in as_completed(futures):
                # 获取 future 的结果，此处为 None，因为 download_image 函数没有返回值
                result = future.result()
                # 更新进度条
                pbar.update()


    # with Pool(processes=max_workers) as p:
    #     with tqdm(total=len(image_urls), desc='total') as pbar:
    #         for i, _ in enumerate(p.imap_unordered(partial_download_image, zip(image_name, image_urls))):
    #         # for i, _ in enumerate(p.imap_unordered(lambda args: download_image(*args), zip(image_name, image_urls, [folder] * len(image_urls)))):
    #             pbar.update()
    #     print("multiprocess_download_images done!")

    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # 提交下载任务
    #     futures = [executor.submit(download_image, name, url, folder) for name, url in zip(image_name, image_urls)]
    #     # 等待所有任务完成
    #     for future in futures:
    #         future.result()

if __name__ == "__main__":
    # 图片链接列表
    with open("mario_laion_image_url/mario-laion-index-url.txt", "r") as f:
        lines = f.readlines()
    image_name = [a.strip().split(" ")[0] for a in lines][1903183:]
    image_urls = [a.strip().split(" ")[1] for a in lines][1903183:]
    # 图片保存目录
    download_folder = "laion-images"
    # 开始下载
    download_images(image_name, image_urls, download_folder, max_workers=64)
