# LocSwinNet

This is a submission to the MLSP 2023 Competion: THE URBAN WIRELESS LOCALIZATION COMPETITION. 
The main task of the competition is to develop highly accurate localization methods which can make use of RSS (pathloss) and ToA measurements.

This repo contains the required files for the reproducibility.


1. Install the requirement:
    ```
    git clone https://github.com/mhh0318/WirelessLocation.git
    cd WirelessLocation
    pip install -r requirements.txt
    ```

2. Download the simulated datasets and their estimate version or noisy version:

    For the `RadioLocSeer/` and `RadioToASeer/LocDBDelay.mat`, you can download from [IEEE Dataport](https://ieee-dataport.org/documents/dataset-pathloss-and-toa-radio-maps-localization-application)

    For the `RadioToASeer/LocDBDelay_Noise20m.mat`, you can download from [LocSwinNet Checkpoint](https://entuedu-my.sharepoint.com/:f:/g/personal/minghui_hu_staff_main_ntu_edu_sg/EktmQSjfvjtNknm5M0Af8FABed7fpVCf7OIDu5KyfMxDaQ?e=6BFwBE) or generate it using `scripts/AddNoiseToTOA.m`.

    After generate ToA Dataset, run the `scripts/toa2img.py` convert the data to image file.

    The `dataset` folder structure is shown below.
    ```
        dataset/
        ├── RadioLocSeer
        │   ├── antennas
        │   ├── gain
        │   ├── mat
        │   ├── png
        │   └── polygon
        ├── RadioToAImage
        │   ├── Est
        │   └── True
        └── RadioToASeer
            ├── LocDBDelay.mat
            └── LocDBDelay_Noise20m.mat
    ```
3. You can train your own model with
    ```
    python main.py --config-path configs --config-name CONFIG_FILE_NAME
    ```

4. You can eval the performance under different situation
    ```
    python eval.py --config-path configs --config-name CONFIG_FILE_NAME
    ```


Here is a summary of evaluation loss (RMSE) meter:

|  ToA + RSS + CityMap   | ToA + RSS  | RSS Only | ToA Only |
|  ----  | ----  | ----  | ----  |
| 2.048  | 1.925 | 5.896 | 2.897 |

# Some Useful Links

[Competition Page](https://urbanlocalizationcompetition.github.io/index.html)

[LocSwinNet Checkpoint](https://entuedu-my.sharepoint.com/:f:/g/personal/minghui_hu_staff_main_ntu_edu_sg/EktmQSjfvjtNknm5M0Af8FABed7fpVCf7OIDu5KyfMxDaQ?e=6BFwBE)
password : default 

[Related Project Page](https://radiomapseer.github.io/LocUNet.html)

This project is heavily based on [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet)