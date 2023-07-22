from Model_64QAM import PHY_Reconstruction_AE#,CSI_Pilot_Features#PHY_Reconstruction_Generator
from Utils_testing import NN_Testing
import tensorflow as tf
if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
    # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
            print(e)

    PHY_Net_gen = PHY_Reconstruction_AE()#PHY_Reconstruction_Generator()
    NN_Testing(PHY_Net_gen, "PHY_dataset_WN64.npz", "PHY_dataset_NoInter_0.8.npz", "logs")#PHY_Net_disc, 
#PHY_dataset_PAYLOADONLYv1_0