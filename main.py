import random  
import features
import data_sets
import modelos
import time


sp1_00 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0000.flac'
sp1_01 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0001.flac'
sp1_02 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0002.flac'
sp1_03 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0003.flac'
sp1_04 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0004.flac'
sp1_05 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0005.flac'
sp1_06 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0006.flac'
sp1_07 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0007.flac'
sp1_08 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0008.flac'
sp1_09 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0009.flac'
sp1_10 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0010.flac'
sp1_11 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0011.flac'
sp1_12 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0012.flac'
sp1_13 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0013.flac'
sp1_14 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0014.flac'
sp1_15 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0015.flac'
sp1_16 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0016.flac'
sp1_17 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0017.flac'
sp1_18 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0018.flac'
sp1_19 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0019.flac'
sp1_20 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0020.flac'
sp1_21 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0021.flac'
sp1_22 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0022.flac'
sp1_23 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0023.flac'
sp1_24 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0024.flac'
sp1_25 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0025.flac'
sp1_26 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0026.flac'
sp1_27 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0027.flac'
sp1_28 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0028.flac'
sp1_29 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0029.flac'
sp1_30 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0030.flac'
sp1_31 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0031.flac'
sp1_32 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0032.flac'
sp1_33 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0033.flac'
sp1_34 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0034.flac'
sp1_35 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0035.flac'
sp1_36 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0036.flac'
sp1_37 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/19/198/19-198-0037.flac'




sp2_00 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0000.flac'
sp2_01 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0001.flac'
sp2_02 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0002.flac'
sp2_03 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0003.flac'
sp2_04 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0004.flac'
sp2_05 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0005.flac'
sp2_06 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0006.flac'
sp2_07 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0007.flac'
sp2_08 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0008.flac'
sp2_09 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0009.flac'
sp2_10 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0010.flac'
sp2_11 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0011.flac'
sp2_12 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0012.flac'
sp2_13 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0013.flac'
sp2_14 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0014.flac'
sp2_15 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0015.flac'
sp2_16 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0016.flac'
sp2_17 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0017.flac'
sp2_18 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0018.flac'
sp2_19 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0019.flac'
sp2_20 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0020.flac'
sp2_21 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0021.flac'
sp2_22 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0022.flac'
sp2_23 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0023.flac'
sp2_24 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0024.flac'
sp2_25 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0025.flac'
sp2_26 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0026.flac'
sp2_27 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0027.flac'
sp2_28 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0028.flac'
sp2_29 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0029.flac'
sp2_30 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0030.flac'
sp2_31 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0031.flac'
sp2_32 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0032.flac'
sp2_33 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0033.flac'
sp2_34 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0034.flac'
sp2_35 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0035.flac'
sp2_36 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0036.flac'
sp2_37 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0037.flac'
sp2_38 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0038.flac'
sp2_39 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0039.flac'
sp2_40 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/26/495/26-495-0040.flac'



sp3_00 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0000.flac'
sp3_01 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0001.flac'
sp3_02 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0002.flac'
sp3_03 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0003.flac'
sp3_04 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0004.flac'
sp3_05 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0005.flac'
sp3_06 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0006.flac'
sp3_07 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0007.flac'
sp3_08 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0008.flac'
sp3_09 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0009.flac'
sp3_10 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0010.flac'
sp3_11 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0011.flac'
sp3_12 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0012.flac'
sp3_13 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0013.flac'
sp3_14 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0014.flac'
sp3_15 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0015.flac'
sp3_16 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0016.flac'
sp3_17 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0017.flac'
sp3_18 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0018.flac'
sp3_19 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0019.flac'
sp3_20 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0020.flac'
sp3_21 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0021.flac'
sp3_22 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0022.flac'
sp3_23 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0023.flac'
sp3_24 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0024.flac'
sp3_25 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0025.flac'
sp3_26 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0026.flac'
sp3_27 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0027.flac'
sp3_28 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0028.flac'
sp3_29 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0029.flac'
sp3_30 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0030.flac'
sp3_31 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0031.flac'
sp3_32 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0032.flac'
sp3_33 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0033.flac'
sp3_34 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0034.flac'
sp3_35 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0035.flac'
sp3_36 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0036.flac'
sp3_37 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0037.flac'
sp3_38 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0038.flac'
sp3_39 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0039.flac'
sp3_40 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/27/123349/27-123349-0040.flac'



sp4_00 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0000.flac'
sp4_01 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0001.flac'
sp4_02 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0002.flac'
sp4_03 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0003.flac'
sp4_04 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0004.flac'
sp4_05 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0005.flac'
sp4_06 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0006.flac'
sp4_07 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0007.flac'
sp4_08 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0008.flac'
sp4_09 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0009.flac'
sp4_10 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0010.flac'
sp4_11 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0011.flac'
sp4_12 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0012.flac'
sp4_13 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0013.flac'
sp4_14 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0014.flac'
sp4_15 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0015.flac'
sp4_16 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0016.flac'
sp4_17 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0017.flac'
sp4_18 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0018.flac'
sp4_19 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0019.flac'
sp4_20 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0020.flac'
sp4_21 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0021.flac'
sp4_22 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0022.flac'
sp4_23 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0023.flac'
sp4_24 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0024.flac'
sp4_25 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0025.flac'
sp4_26 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0026.flac'
sp4_27 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0027.flac'
sp4_28 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0028.flac'
sp4_29 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0029.flac'
sp4_30 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0030.flac'
sp4_31 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0031.flac'
sp4_32 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0032.flac'
sp4_33 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0033.flac'
sp4_34 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0034.flac'
sp4_35 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0035.flac'
sp4_36 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0036.flac'
sp4_37 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0037.flac'
sp4_38 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0038.flac'
sp4_39 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0039.flac'
sp4_40 = '/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/32/4137/32-4137-0040.flac'


user01 = [sp1_00,sp1_01,sp1_02,sp1_03,sp1_04,sp1_05,sp1_06,sp1_07,sp1_08,sp1_09,sp1_10,sp1_11,sp1_12,sp1_13,sp1_14,sp1_15,sp1_16,sp1_17,sp1_18,sp1_19,
				sp1_20,sp1_21,sp1_22,sp1_23,sp1_24,sp1_25,sp1_26,sp1_27,sp1_28,sp1_29,sp1_30,sp1_31,sp1_32,sp1_33,sp1_34,sp1_35,sp1_36,sp1_37]
user02 = [sp2_00,sp2_01,sp2_02,sp2_03,sp2_04,sp2_05,sp2_06,sp2_07,sp2_08,sp2_09,sp2_10,sp2_11,sp2_12,sp2_13,sp2_14,sp2_15,sp2_16,sp2_17,sp2_18,sp2_19,
				sp2_20,sp2_21,sp2_22,sp2_23,sp2_24,sp2_25,sp2_26,sp2_27,sp2_28,sp2_29,sp2_30,sp2_31,sp2_32,sp2_33,sp2_34,sp2_35,sp2_36,sp2_37,sp2_38,sp2_39,sp2_40]
user03 = [sp3_00,sp3_01,sp3_02,sp3_03,sp3_04,sp3_05,sp3_06,sp3_07,sp3_08,sp3_09,sp3_10,sp3_11,sp3_12,sp3_13,sp3_14,sp3_15,sp3_16,sp3_17,sp3_18,sp3_19,
				sp3_20,sp3_21,sp3_22,sp3_23,sp3_24,sp3_25,sp3_26,sp3_27,sp3_28,sp3_29,sp3_30,sp3_31,sp3_32,sp3_33,sp3_34,sp3_35,sp3_36,sp3_37,sp3_38,sp3_39,sp3_40]
user04 = [sp4_00,sp4_01,sp4_02,sp4_03,sp4_04,sp4_05,sp4_06,sp4_07,sp4_08,sp4_09,sp4_10,sp4_11,sp4_12,sp4_13,sp4_14,sp4_15,sp4_16,sp4_17,sp4_18,sp4_19,
				sp4_20,sp4_21,sp4_22,sp4_23,sp4_24,sp4_25,sp4_26,sp4_27,sp4_28,sp4_29,sp4_30,sp4_31,sp4_32,sp4_33,sp4_34,sp4_35,sp4_36,sp4_37,sp4_38,sp4_39,sp4_40]




corte = 25
fin =   30
user01_train = user01[:corte]
user02_train = user02[:corte]
user03_train = user03[:corte]
user04_train = user04[:corte]

user01_test = user01[corte:fin]
user02_test = user02[corte:fin]
user03_test = user03[corte:fin]
user04_test = user04[corte:fin]

train_path_list = [user01_train,user02_train,user03_train,user04_train]
test_path_list  = [user01_test,user02_test,user03_test,user04_test]
file_class_list  = [0,1,2,3]


txt_file_path = '/home/ar/Tesis_Pedro/txt/rendimiento.txt'
#modelos.create_txt_all_file(txt_file_path)



#SVM parametros
kernel = 'poly'
C      = 0.1
gamma  = 0.1
tol    = 0.001
degree = 3
coef0  = 0

#LDA parametros
kernel     = ''
reg_param  = 0.25
tol        = 0.001

#SVM parametros
kernel            = 'gini'
min_samples_split = 2
min_samples_leaf  = 1

svm_param_list   = [C, gamma, tol, degree, coef0]
lda_param_list   = [reg_param, tol]
dt_param_list    = [min_samples_split,min_samples_leaf]


num_iterations = 30
for iteration in range(0, num_iterations):
    start = time.time()
    print('iteracion ' + str(iteration) + ' de ' + str(num_iterations))
	
    """
    if(iteration%2==0):
        FEATURE = 'mfb'
    else:
        FEATURE = 'mfcc'
    """
    FEATURE = 'lpc'

    pre_emph_coeff  = 0.85 + random.randint(0,14)*0.01
    frame_size_time = 0.5  + random.randint(0,10)*0.1
    frame_hop_time  = 0.05 + random.randint(0,11)*0.01

    if(FEATURE == 'mfb'):
        #MFB parametros
        nfilt           = random.randint(25,50)
		
	
        param_list  = [pre_emph_coeff, frame_size_time, frame_hop_time, nfilt]
        print('pre_emph = ' + str(param_list[0]) + ', ' + 'window = ' + str(param_list[1]) + ', ' + 'hop = ' + str(param_list[2]) + ', ' + 'nfilt = ' + 	str(param_list[3]))


    elif(FEATURE == 'mfcc'):
        #MFCC parametros
        nfilt           = random.randint(25,50)
        num_ceps        = random.randint(12,nfilt-1)
        cep_lifter      = random.randint(0,20)
        deltas          = random.randint(0,2)
	
        param_list  = [pre_emph_coeff, frame_size_time, frame_hop_time, nfilt, num_ceps, cep_lifter, deltas]
        print('pre_emph = ' + str(param_list[0]) + ', ' + 'window = ' + str(param_list[1]) + ', ' + 'hop = ' + str(param_list[2]) + ', ' + 'nfilt = ' + str(param_list[3]) + ', ' + 'ceps = ' + str(param_list[4]) + ', ' + 'lift = ' + str(param_list[5]) + ', ' + 'deltas = ' + str(param_list[06]))


    elif(FEATURE == 'lpc'):
        #LPC parametros
        order           = random.randint(5,20)
        gain            = random.randint(0,1)
	
        param_list   = [pre_emph_coeff, frame_size_time, frame_hop_time, order, gain]
        print('pre_emph = ' + str(param_list[0]) + ', ' + 'window = ' + str(param_list[1]) + ', ' + 'hop = ' + str(param_list[2]) + ', ' + 'order = ' + str(param_list[3]) + ', ' + 'gain = ' + str(param_list[4]))
	
	
    modelos.run_model(train_path_list, test_path_list, param_list, txt_file_path, FEATURE)
    end                 = time.time()
    iteration_time       = str(end - start)

    print('iteration time = ' + str(iteration_time))
    print('******************************************************************')


