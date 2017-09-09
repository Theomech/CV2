import landmarks, procrustes, radiograph, pca

# here we call all functions

# 1.1 Load the provided landmarks into your program
landmarks = landmarks.loadLandmarks()  # this will return a unified 28x8x40x2 matrix for all landmarks (original and mirrored)

# 1.2 Pre-process the landmarks to normalize translation, rotation and scale differences (Procrustes Analysis)
Z = procrustes.proc(landmarks)

# 1.3. Analyze the data using a Principal Component Analysis (PCA), exposing shape class variations
ASMeigenval, ASMeigenvec, ASMmu = pca.pcaLand(Z)

# 1.4. Analyze the obtained principal components

# 2. Pre-process the dental radiographs
preprocessedRadios = radiograph.getPreprocessedRadios()

