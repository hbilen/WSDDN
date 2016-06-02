# Weakly Supervised Deep Detection Networks (WSDDN)



## Citing WSDDN
If you find the code useful, please cite:

```latex
    @inproceedings{Bilen16,
      author     = "Bilen, H. and Vedaldi, A.",
      title      = "Weakly Supervised Deep Detection Networks',
      booktitle  = "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition',
      year       = "2016"
    }
```

## Installation
1. Clone WSDDN repository:

    ```Shell
    git clone --recursive  https://github.com/hbilen/WSDDN
    ```
    
2. Compile matconvnet toolbox using `cnn_wsddn_compile`: (see the options in [http://www.vlfeat.org/matconvnet/install/](http://www.vlfeat.org/matconvnet/install/)

3. If you want to train or test on the PASCAL VOC,

    a.  Download the PASCAL VOC 2007 devkit and dataset [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/) under `data` folder

    b.  Download the pre-computed edge-boxes from the links below (for trainval and test splits):

      [https://drive.google.com/open?id=0B0evBVYO74MENXZCWnZmT2kyUEE](https://drive.google.com/open?id=0B0evBVYO74MENXZCWnZmT2kyUEE)
      
      [https://drive.google.com/open?id=0B0evBVYO74MEMUluNm4tamEyMHM](https://drive.google.com/open?id=0B0evBVYO74MEMUluNm4tamEyMHM)

    c. Download the pre-trained network (VGGF-EB-BoxSc-SpReg). Note that it gives slightly better performance reported than in the paper (35.3% mAP instead of 34.5% mAP)

      [https://drive.google.com/open?id=0B0evBVYO74MEdjJVR19URUNFOGc](https://drive.google.com/open?id=0B0evBVYO74MEdjJVR19URUNFOGc)


## Demo

After completing the installation and downloading the required files, you are ready for the demo

```matlab
            cd scripts;
            opts.modelPath = '....' ;
            opts.imdbPath = '....' ;
            opts.train.gpus = .... ;
            cnn_wsddn_demo(opts) ;
                        
```

## Test

```matlab
            addpath scripts;
            opts.modelPath = '....' ;
            opts.imdbPath = '....' ;
            opts.train.gpus = .... ;
            cnn_wsddn_test(opts) ;
                        
```

## Train

Download an ImageNet pre-trained model from [http://www.vlfeat.org/matconvnet/pretrained/](http://www.vlfeat.org/matconvnet/pretrained/)

```matlab
            addpath scripts;
            opts.modelPath = '....' ;
            opts.imdbPath = '....' ;
            opts.train.gpus = .... ;
            [net,info] = cnn_wsddn_train(opts) ;
                        
```


### License
The analysis work performed with the program(s) must be non-proprietary work. Licensee and its contract users must be or be affiliated with an academic facility. Licensee may additionally permit individuals who are students at such academic facility to access and use the program(s). Such students will be considered contract users of licensee. The program(s) may not be used for commercial competitive analysis (such as benchmarking) or for any commercial activity, including consulting.
