# speech_features
Medición del rendimiento de características de la señal de voz vs clasificadores para el reconocimiento de locutor.

Características de la señal de voz:
  MFB     Mel filter bank
  MFCC    Mel frequency cepstral coefficients
  LPC     Linear predictive coefficients
  
Clasificadores:
  SVM     Support vector machines
  LDA     Linear discriminant analisys
  DT      Desition tree classifier
  
  
Se extraen las caracteristicas de la señal de voz (utilizando varios parámetros en el cálculo de las características) se entrenan los modelos. Posteriormente se mide el rendimiento de los modelos con el conjunto de prueba.

Medidas de rendimiento:
  Accuracy
  Precision
  Recall
  F1
