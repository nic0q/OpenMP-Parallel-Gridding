all: gridding

gridding:
	gcc -Ofast -lm -fopenmp gridding.c -o gridding

clean:
	rm -f *.o *.exe gridding

test:
	./gridding -i hltau_completo_uv.csv -o datosgrideados_shared -d 0.003 -N 2048 -c 100000 -t 10