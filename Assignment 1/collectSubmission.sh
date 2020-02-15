read -p "Enter your First Name :" FIRST_NAME
read -p  "Enter your Last Name :" LAST_NAME
jupyter nbconvert svm.ipynb --output "${LAST_NAME}_1.html"
jupyter nbconvert softmax.ipynb --output "${LAST_NAME}_2.html"
jupyter nbconvert two_layer_net.ipynb --output "${LAST_NAME}_3.html"
zip -m "${LAST_NAME}_${FIRST_NAME}_assignment1.zip" *.html
zip -u "${LAST_NAME}_${FIRST_NAME}_assignment1.zip" cs231n/*.py cs231n/classifiers/*
zip -u "${LAST_NAME}_${FIRST_NAME}_assignment1.zip" *.ipynb