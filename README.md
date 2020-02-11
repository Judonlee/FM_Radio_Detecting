# FM_Radio_Detecting

**Structure of FM Broadcast Signal Detection Project**:

    ├── 01.getFeatures
    |   |—— features               // Save extracted features
    |   |—— makeLogFbanks.py       // Extracting audio acoustic features
    |   └── run.sh                 // The sh script used to run the features extraction .py file.
    ├── 02.trainModel         
    │   ├── model                  // save the trained model. 
    │   └── train_cnn     
    |       ├── net_component.py   // the neural networks.
    |       ├── make_torchdata.py  // Prepare network input data.
    |       ├── train.py           // Train the model. 
    |       └── run.sh             // The sh script used to run train.py file.  
    ├── 03.inference 
    |   ├── result                 // Visualization of test data.   
    |   ├── net_component.py       // define neural networks.
    |   ├── make_torchdata.py      //  Prepare network input data.
    |   ├── inference.py           // Load the model for testing
    |   └── run.sh                 // The sh script used to run inference.py file.
    ├── 04.inference 
    |   ├── feature                // Save the extracted features. 
    |   ├── net_component.py       // Define neural networks
    |   ├── make_torchdata.py      // Prepare network input data. 
    |   └── fm_predict.py          // Detection of FM audio files
    ├── 05.S2T_Baidu 
    |   ├── speech_signal          // Save file detected as speech in step 04. 
    |   └── Baidu_speech2text.py   // Speech-to-text conversion for audio files
    ├── data_fm                    // Example of Fm band
    |—— dataset                    // Training data
    |—— doc                        
    |── README.md
    └── requirements.txt             
  
**System block diagram**：
![Picture](./docs/1.png)

## Dependencies

**python3.6** + **Pytorch** + **baidu-aip** ：

* numpy==1.15.1, scipy==1.1.0, matplotlib==2.2.3, tqdm==4.26.0
* python-speech-features==0.6, sox==1.3.3
* scikit-learn==0.19.2, scikits.talkbox==0.2.5 
* torch==0.4.1, torchvision==0.2.1
* baidu-aip==2.2.13.0

Installation: pip install -r requirements.txt

## Dataset
link:https://pan.baidu.com/s/1KtqtVt8KHN0R_74fYrGydw code：o3wf

## Audio features extracted
FBank、MFCC、SDC、PLP、Spectrum...

## Training and Evaluation

### Training

**Parameters**：

    num_classes = 2
    num_epochs = 10
    batch_size = 120
    learning_rate = 0.001
        
optimization：

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### Evaluate
acc：**99.83%** 

sklearn.manifold t-SNE ：

    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(last_embedding)

![2D fig](./docs/3.gif)

![3D fig](./docs/4.gif)


## Inference
load model

    # model = torch.load('../02.trainModel/model/model_speech_noise_FBank_9.pth')
    model = torch.load('../02.trainModel/model/model_speech_noise_FBank_9.pth', map_location='cpu')

## FM Predicting

    wavpath = '../data_fm/'
    datalist = 'data_list.txt'
    data_dir = "./features/LogFbank/"
    features_list = 'data_list_lfb.txt'

    result_dic = main(wavpath, datalist, data_dir, features_list)

    print('\nresult_dic: {}\n'.format(result_dic))

    list1 = sorted(result_dic.items(), key=lambda x: x[0])
    for idx, item in enumerate(list1):
        # print(item)
        item = list(item)
        print('Freq： {}, signal label： {}, Prob： {}\n'.format(item[0], item[1][0], item[1][1][1]))

        if item[1][0] == 1:
            oldpath = os.path.join('../data_fm/2019-01-02_16_05_36', str(item[0]) + 'MHz.wav')
            newpatn = os.path.join('../05.S2T_Baidu/speech_signal', str(item[0]) + 'MHz.wav')
            shutil.copy(oldpath, newpatn)
    
    output：
         
    result_dic: {'91.1': [0, [0.99989545, 0.000104586456]], '91.2': [0, [0.99985456, 0.00014543523]], '91.3': [1, [9.564074e-05, 0.9999044]], '91.4': [0, [0.9998274, 0.00017262818]], '91.5': [0, [0.99983656, 0.00016338678]], '91.6': [0, [0.99991846, 8.157516e-05]], '91.7': [0, [0.9999403, 5.973239e-05]], '91.8': [1, [8.419428e-05, 0.99991584]], '91.9': [0, [0.99991417, 8.585341e-05]], '91': [0, [0.999445, 0.00055502006]], '92.0': [0, [0.99994874, 5.1229697e-05]], '92.1': [0, [0.999933, 6.6949084e-05]], '92.2': [0, [0.99823713, 0.0017629083]], '92.3': [1, [0.00011307727, 0.9998869]], '92.4': [0, [0.9987589, 0.0012410908]], '92.5': [1, [0.0065199328, 0.9934801]], '92.6': [0, [0.6229184, 0.3770815]], '92.7': [0, [0.9963198, 0.003680264]], '92.8': [0, [0.99994874, 5.121509e-05]], '92.9': [0, [0.9999486, 5.1379164e-05]], '93.0': [1, [9.812242e-05, 0.9999019]], '93.1': [0, [0.99992263, 7.738675e-05]], '93.2': [0, [0.999925, 7.494193e-05]], '93.3': [0, [0.9999572, 4.279748e-05]], '93.4': [0, [0.9999591, 4.0913146e-05]], '93.5': [1, [5.420044e-05, 0.99994576]], '93.6': [0, [0.99996376, 3.6231482e-05]], '93.7': [0, [0.98870856, 0.011291448]], '93.8': [0, [0.99995506, 4.488294e-05]], '93.9': [1, [0.41586345, 0.58413655]], '94.0': [0, [0.9999702, 2.980607e-05]], '94.1': [0, [0.99997425, 2.5763024e-05]], '94.2': [0, [0.9999751, 2.4864961e-05]], '94.3': [0, [0.9999701, 2.9962186e-05]], '94.4': [0, [0.9999715, 2.8433105e-05]], '94.5': [0, [0.9999267, 7.333816e-05]], '94.6': [0, [0.9999354, 6.4645574e-05]], '94.7': [0, [0.9999709, 2.9125797e-05]], '94.8': [0, [0.99996614, 3.3841497e-05]], '94.9': [1, [0.000115533556, 0.9998845]], '95.0': [0, [0.9999559, 4.4131415e-05]], '95.1': [0, [0.99996257, 3.7393598e-05]], '95.2': [0, [0.9999695, 3.0536925e-05]], '95.3': [0, [0.9999604, 3.9615235e-05]], '95.4': [1, [6.747619e-05, 0.9999325]], '95.5': [0, [0.9999349, 6.513262e-05]], '95.6': [0, [0.9999579, 4.210114e-05]], '95.7': [0, [0.9999645, 3.5511384e-05]], '95.8': [0, [0.999956, 4.392865e-05]], '95.9': [1, [0.00074070145, 0.9992593]], '96.0': [0, [0.9998944, 0.000105659245]]}

## TO Do
- [1]

## Reference
[1]
