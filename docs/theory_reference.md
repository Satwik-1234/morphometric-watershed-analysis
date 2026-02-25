# Theory & Formula Reference

Complete mathematical reference for all morphometric parameters implemented in this pipeline.

---

## 1. Linear Aspects

### Stream Number Law (Horton, 1945)
$$N_u = a \cdot R_b^{-u}$$

Where $R_b$ is the bifurcation ratio and $u$ is stream order.  
**Regression:** Plot $\log(N_u)$ vs $u$ — slope = $-\log(R_b)$

### Stream Length Law (Horton, 1945)
$$\bar{L}_u = \bar{L}_1 \cdot R_L^{u-1}$$

Where $R_L$ is the stream length ratio.

### Weighted Mean Bifurcation Ratio (Strahler, 1957)
$$wR_{bm} = \frac{\sum (R_b \cdot (N_u + N_{u+1}))}{\sum (N_u + N_{u+1})}$$

---

## 2. Areal Aspects

### Drainage Density (Horton, 1945)
$$D_d = \frac{\sum L_u}{A} \quad [\text{km/km}^2]$$

### Elongation Ratio (Schumm, 1956)
$$R_e = \frac{2}{L_b} \sqrt{\frac{A}{\pi}}$$

Range: circular (1.0) → elongated (0.0)

### Circularity Ratio (Miller, 1953)
$$R_c = \frac{4\pi A}{P^2}$$

Range: 0 → 1 (perfect circle = 1)

### Form Factor (Horton, 1932)
$$F_f = \frac{A}{L_b^2}$$

---

## 3. Relief Aspects

### Hypsometric Integral (Strahler, 1952)
$$HI = \frac{\bar{H} - H_{min}}{H_{max} - H_{min}}$$

Stages: Monadnock (HI > 0.6) · Mature (0.35–0.6) · Peneplain (< 0.35)

### Ruggedness Number (Strahler, 1958)
$$R_n = H \times D_d$$

### Terrain Ruggedness Index (Riley et al., 1999)
$$TRI = \sqrt{\sum_{i=1}^{8}(x_i - x_0)^2}$$

---

## 4. Tectonic Indices

### IAT Classification (El Hamdouni et al., 2008)
$$IAT = \frac{Score_{AF} + Score_T + Score_{Vf} + Score_{Smf}}{4}$$

Scores: 1 = high activity · 2 = moderate · 3 = low

| IAT | Class |
|-----|-------|
| 1.0–1.5 | Class 1 — Very High |
| 1.5–2.0 | Class 2 — High |
| 2.0–2.5 | Class 3 — Moderate |
| 2.5–3.0 | Class 4 — Low |

---

## 5. Channel Steepness

### Normalised Steepness Index
$$k_{sn} = S \cdot A^{\theta_{ref}} \quad (\theta_{ref} = 0.45)$$

### Chi Coordinate (Perron & Royden, 2012)
$$\chi = \int_{x_b}^{x} \left(\frac{A_0}{A(x')}\right)^m dx' \quad (m = 0.45, \; A_0 = 1 \text{ m}^2)$$

---

## 6. Flood Hazard

### TWI (Beven & Kirkby, 1979)
$$TWI = \ln\left(\frac{A_s}{\tan\beta}\right)$$

### SPI (Moore et al., 1991)
$$SPI = A_s \cdot \tan\beta$$

### STI (Moore & Burch, 1986)
$$STI = \left(\frac{A_s}{22.13}\right)^{0.6} \cdot \left(\frac{\sin\beta}{0.0896}\right)^{1.3}$$

### FFPI (Smith, 2003 — adapted)
$$FFPI = 0.35 \cdot S_{norm} + 0.25 \cdot R_{norm} + 0.25 \cdot TWI_{norm} + 0.15 \cdot SPI_{norm}$$

