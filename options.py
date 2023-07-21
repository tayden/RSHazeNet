
class Options():
    def __init__(self):
        super().__init__()

        self.Epoch = 1000
        self.Learning_Rate = 2e-4
        self.Batch_Size_Train = 14
        self.Batch_Size_Val = 14
        self.Patch_Size_Train = 512
        self.Patch_Size_Val = 512
        # training
        self.Input_Path_Train = 'I://remote_sensing_haze_dataset/Haze1k_thick/train/input/'
        self.Target_Path_Train = 'I://remote_sensing_haze_dataset/Haze1k_thick/train/target/'
        # validation
        self.Input_Path_Val = 'I://remote_sensing_haze_dataset/Haze1k_thick/test/input/'
        self.Target_Path_Val = 'I://remote_sensing_haze_dataset/Haze1k_thick/test/target/'
        # testing
        self.Input_Path_Test = 'I://remote_sensing_haze_dataset/Haze1k_thick/test/input/'
        self.Target_Path_Test = 'I://remote_sensing_haze_dataset/Haze1k_thick/test/target/'
        self.Result_Path_Test = 'I://remote_sensing_haze_dataset/Haze1k_thick/test/result/'

        self.MODEL_SAVE_PATH = './'
        self.MODEL_PRE_PATH = './model_best.pth'

        self.Num_Works = 4
        self.CUDA_USE = True