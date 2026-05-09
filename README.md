# Learning-Probability-Density-Functions-using-data_UCS654
# Probability Density Function Estimation Using Data

## 1. Methodology
Data Collection → Column Selection → Data Transformation → PDF Estimation using GAN → PDF Approximation from Samples → PDF Visualization

---

## 2. Description
- **Dataset:** `data.csv`  
- **Target Column:** `no2`  
- **Transformation Applied:**  
  z = T(x) = x + arcsin(bx)  
- **PDF Estimation Method:** Generative Adversarial Network (GAN)  
- **Final Output:** Visualization of the estimated PDF  

---

## 3. Objectives
- Perform data transformation  
- Estimate the probability density function (PDF)  
- Approximate the PDF from generated samples  
- Visualize the resulting distribution  

---

## 4. Results

### 4.1 Transformation Parameters

| Parameter | Value |
|----------|-------|
| a_r      | 0.5   |
| b_r      | 0.9   |

---

### 4.2 GAN Training Log

| Epoch   | D Loss | G Loss |
|--------|--------|--------|
| 0/500   | 1.3473 | 0.7359 |
| 50/500  | 1.3764 | 0.7055 |
| 100/500 | 1.3900 | 0.6894 |
| 150/500 | 1.3838 | 0.6953 |
| 200/500 | 1.3942 | 0.6943 |
| 250/500 | 1.3865 | 0.6982 |
| 300/500 | 1.3914 | 0.6932 |
| 350/500 | 1.3874 | 0.6843 |
| 400/500 | 1.3877 | 0.6898 |
| 450/500 | 1.3851 | 0.6896 |

**GAN training completed successfully.**

---

### 4.4 Observations

**1. Mode Coverage:**  
The GAN effectively captures multiple modes introduced by the nonlinear sine-based transformation from x to z.

**2. Training Stability:**  
After initial fluctuations, training stabilizes without evidence of mode collapse, indicating a well-balanced generator and discriminator.

**3. Distribution Quality:**  
The KDE and histogram-based PDFs closely align, suggesting that the generator has learned a reliable approximation of the underlying distribution.
