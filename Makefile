all: gridding sec test

gridding:
	gcc gridding.c -o gridding -fopenmp -lm

sec:
	gcc sec.c -o sec -lm

clean:
	rm -f *.o *.exe gridding sec

test:
	./gridding -i hltau_completo_uv.csv -o datosgrideados -d 0.003 -N 2048 -c 700000 -t 5
	./sec
