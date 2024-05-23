# DL Model Benchmarking for WiFi Sensing

## Project Overview
This project focuses on benchmarking deep learning models for WiFi sensing. Conducted between September 2023 and December 2023, it includes in-depth evaluations and performance analysis on substantial datasets.

## Technologies Used
- **Wireless Networks**
- **Deep Learning**
- **Python**
- **PyTorch**

## Deep Learning Models
The models benchmarked in this project include:
- BiLSTM (done)
- CNN+GRU (done)
- ResNET101 (done) + extra (Resnet50)
- MLP (done)
- LeNET (done)

## Datasets

### 1. UT-HAR
UT-HAR is the first public CSI dataset for human activity recognition. It was collected using an Intel 5300 NIC with three pairs of antennas that each record 30 subcarriers and comprises seven categories. The data is segmented using a sliding window in accordance with previous works, resulting in roughly 5,000 samples in total.

- **Platform:** Intel 5300 NIC
- **CSI size:** 1 x 250 x 90
- **Number of classes:** 7
- **Classes:** lie down, fall, walk, pickup, run, sit down, stand up
- **Train number:** 3977
- **Test number:** 996

### 2. NTU-HAR
This dataset was collected using a new platform based on the Atheros CSI tool, enabling CSI extraction of 114 subcarriers of 40Hz directly to IoT devices. It includes Human Activity Recognition (HAR) tasks.

- **Platform:** Atheros CSI tool
- **CSI size:** 3 x 114 x 500 (antenna, subcarrier, packet)
- **Number of classes:** 6
- **Classes:** box, circle, clean, fall, run, walk
- **Train number:** 936
- **Test number:** 264

### 3. NTU-HumanID
This dataset includes Human Identification (HumanID) tasks. The gaits of 14 subjects are present in this dataset.

- **Platform:** Atheros CSI tool
- **CSI size:** 3 x 114 x 500 (antenna, subcarrier, packet)
- **Number of classes:** 14
- **Classes:** gaits of 14 subjects
- **Train number:** 546
- **Test number:** 294

### 4. Widar
Widar is the largest WiFi sensing dataset for gesture recognition, comprising 22 categories and 43K samples. The data is processed and transformed to the body-coordinate velocity profile (BVP) to remove environmental dependencies.

- **Platform:** Intel 5300 NIC
- **BVP size:** 22 x 20 x 20 (time, x velocity, y velocity)
- **Number of classes:** 22
- **Classes:** Push&Pull, Sweep, Clap, Slide, Draw-N(H), Draw-O(H), Draw-Rectangle(H), Draw-Triangle(H), Draw-Zigzag(H), Draw-Zigzag(V), Draw-N(V), Draw-O(V), Draw-1, Draw-2, Draw-3, Draw-4, Draw-5, Draw-6, Draw-7, Draw-8, Draw-9, Draw-10
- **Train number:** 34926
- **Test number:** 8726

## Evaluation Methodology
The evaluation process involved several steps:
- **Data Preprocessing:** Detailed steps on how data was cleaned and prepared for model training.
- **Model Selection:** Criteria for selecting the deep learning models used in this study.
- **Training and Validation:** Procedures and hyperparameters used for training the models.
- **Performance Metrics:** Metrics used to evaluate model performance, such as accuracy, precision, recall, F1-score, etc.

## Parameters Calculated
The following parameters were calculated for each model on each dataset:
- Training accuracy
- Testing accuracy
- Validation accuracy
- Training time
- Inference time
- Memory requirement for training a particular model
- Number of rows
- Number of features of training and testing data

## Results and Analysis
In our comparative analysis, we observed that simple models like LeNet and MLP perform
 nearly as well as more complex models such as BiLSTM, ResNet101, and CNN+GRU across
 datasets NTU-Fi_HAR, NTU-Fi-HumanID, and UT_HAR. When evaluating efficiency in terms of
 inference time and resource utilization (RAM, GPU memory), it becomes evident that the
 straightforward architectures of LeNet and MLP outperform the others in the mentioned three
 datasets.
 However, for the Widar dataset, our findings indicate that a more powerful model is necessary to
 achieve higher accuracy. This suggests that dataset-specific characteristics may require a
 different approach, emphasizing the importance of tailoring model selection to the unique
 demands of each dataset.

## Conclusion
 In wrapping up our study, we've learned a lot about how different types of models
 perform on various datasets. Surprisingly, simple models like LeNet and MLP did really
 well compared to fancier ones like BiLSTM, ResNet101, and CNN+GRU in most cases.
 These simpler models also turned out to be more efficient in terms of how quickly they
 make predictions and how much computer resources they need.
 However, there was a twist when we looked at the Widar dataset. Here, we discovered
 that to get higher accuracy, we need to use more powerful and complex models. So, it
 seems like the best choice depends on the specific dataset we're working with.
 In a nutshell, our benchmark model, with its mix of different kinds of models, can handle
 a variety of tasks quite well. The key takeaway is that picking the right model depends
 on the job at hand– sometimes simple is best, but other times, we might need to go for
 a more powerful approach


## Getting Started
To get started with this project, follow these steps:

1. **Clone the repository**
    ```sh
    git clone https://github.com/yourusername/DL-Model-Benchmarking-WiFi-Sensing.git
    cd DL-Model-Benchmarking-WiFi-Sensing
    ```

2. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

3. **To Run go to WiFi-CSI-Sensing-Benchmark and Read Intruction**
    ```sh
    cd "WiFi-CSI-Sensing-Benchmark"
    ```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or inquiries, please contact [dheerajdeshwal9811@gmail.com](mailto:dheerajdeshwal9811@gmail.com).

---

Happy Benchmarking! 🚀

---

