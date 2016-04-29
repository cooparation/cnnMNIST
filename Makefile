#CNN test
gcc=g++
CCFLAGS=-g -O0
bin=main.exe
obj:=startCNN.o

${bin}:${obj}
	${gcc} ${CCFLAGS} -o ${bin} ${obj}
${obj}:startCNN.cpp cnn.cpp cnn.h global.h ReadData.h
	${gcc} ${CCFLAGS} -c startCNN.cpp cnn.cpp

clean:
	-rm *.o *.exe
