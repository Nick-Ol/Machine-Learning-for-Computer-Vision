if ispc
    mex -O -largeArrayDims -c svm.cpp
    mex -O -largeArrayDims -c svm_model_matlab.c
    mex -O -largeArrayDims svmtrain_libsvm.c svm.obj svm_model_matlab.obj
    mex -O -largeArrayDims svmpredict_libsvm.c svm.obj svm_model_matlab.obj
    mex -O -largeArrayDims read_sparse.c
else
    mex -O -largeArrayDims -c svm.cpp  
    mex -O -largeArrayDims -c svm_model_matlab.c 
    mex -O -largeArrayDims svmtrain_libsvm.c svm.o svm_model_matlab.o 
    mex -O -largeArrayDims svmpredict_libsvm.c svm.o svm_model_matlab.o 
    mex -O -largeArrayDims read_sparse.c
end