% Leer los datos de los archivos
s1_s = fread(fopen("datosgrideados_sharedr.raw", "r"), "float");
s2_s = fread(fopen("datosgrideados_sharedi.raw", "r"), "float");

s1_p = fread(fopen("datosgrideados_privater.raw", "r"), "float");
s2_p = fread(fopen("datosgrideados_privatei.raw", "r"), "float");

% Reshape de los datos
re_s = reshape(s1_s, 2048, 2048);
re_p = reshape(s1_p, 2048, 2048);
im_s = reshape(s2_s, 2048, 2048);
im_p = reshape(s2_p, 2048, 2048);

% Crear números complejos
v_s = complex(re_s, im_s);
v_p = complex(re_p, im_p);

% Calcular transformada de Fourier e inversa
I_s = fftshift(ifft2(v_s));
I_p = fftshift(ifft2(v_p));

% Crear una figura con dos imágenes en una fila
figure;

% Primera imagen
subplot(1,2,1);
imagesc(abs(I_s));
colormap('hot');
title('Shared matrix result image');

% Segunda imagen
subplot(1,2,2);
imagesc(abs(I_p));
colormap('hot');
title('Private matrix result image');