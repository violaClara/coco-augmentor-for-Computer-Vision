from setuptools import setup, find_packages

setup(
    name="coco-augmentor-vioo",  # Nama paket
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,   # Penting biar file HTML kebawa
    install_requires=[
        "flask",
        "albumentations",
        "opencv-python-headless",
        "numpy"
    ],
    entry_points={
        'console_scripts': [
            # FORMAT: nama-perintah = nama_folder.nama_file:nama_fungsi
            'coco-augmentor-vioo=coco_augmentor.app:run_app', 
        ],
    },
)