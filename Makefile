all: gridding sec

gridding:
	gcc gridding.c -fopenmp -lm -o gridding

sec:
	gcc sec.c -o sec -lm

clean:
	rm -f *.o *.exe gridding sec

test:
	./gridding -i hltau_completo_uv.csv -o datosgrideados -d 0.003 -N 2048 -c 300000 -t 10