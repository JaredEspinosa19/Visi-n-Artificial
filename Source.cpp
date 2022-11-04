#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string.h>
#include <iostream>
#include <math.h>

# define M_PI  3.14159265358979323846

using namespace cv;
using namespace std;

double applyFilterSobel(Mat resize_image, Mat Kernel, int n, int x, int y) {
	//Crear un recorrido del tamaño n
	int dim = (n - 1) / 2;
	int k = 0, l = 0;
	int j = 0, i = 0;

	//operadores
	double acc = 0;
	float v_kernel = 0;
	double v_image = 0;
	//float average = 0;
	//posicion
	int x_n = x + dim;
	int y_n = y + dim;

	for (int k = -dim; k <= dim; k++) {
		j = 0;
		for (int l = -dim; l <= dim; l++) {
			v_kernel = Kernel.at<float>(Point(j, i));
			v_image = static_cast<float>(resize_image.at<uchar>((y_n + k), (x_n + l)));
			acc = acc + (v_kernel * v_image);
			j++;
		}
		i++;
	}

	return abs(acc);
}

double applyFilter(Mat resize_image, Mat Kernel, int n, int x, int y) {

	//Crear un recorrido del tamaño n
	int dim = (n - 1) / 2;
	int k = 0, l = 0;
	int j = 0, i = 0;

	//operadores
	double acc = 0;
	float v_kernel = 0;
	double v_image = 0;
	float average = 0;
	//posicion
	int x_n = x + dim;
	int y_n = y + dim;

	//cout << "Pixel imagen original (posicion) (x,y): " << x << y << endl;
	//cout << "Posicion relativas de la nueva imagen (x,y) " << x_n << y_n << endl;

	for (int k = -dim; k <= dim; k++) {
		j = 0;
		for (int l = -dim; l <= dim; l++) {
			v_kernel = Kernel.at<float>(Point(j, i));

			v_image = static_cast<float>(resize_image.at<uchar>((y_n + k), (x_n + l)));
			//cout << (v_kernel * v_image) <<" Resultado multiploicacion en x:" << j << "y:" <<i << endl;
			acc = acc + (v_kernel * v_image);
			//cout << v_kernel * v_image << endl;
			average = average + v_kernel;
			j++;
		}
		i++;
	}

	//cout << acc << "sdasd " << abs(acc / (n * n)) << endl;
	return abs(acc / average);

}

Mat gaussFilter(Mat resize_image, Mat Kernel, int n) {

	int dim = (n - 1) / 2;
	int cols = resize_image.cols - (n - 1);
	int rows = resize_image.cols - (n - 1);


	//Crear imagen
	Mat main_image = Mat::zeros(rows, cols, CV_8UC1);

	//hacer un recorrido
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double n_value = applyFilter(resize_image, Kernel, n, j, i);
			main_image.at<uchar>(Point(j, i)) = uchar(n_value);
		}
	}
	//

	return main_image;
}


void createExcel(Mat image) {

	FILE* excel_image;
	excel_image = fopen("resize_Image.xls", "w");

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			double val_orig = image.at<uchar>(Point(j, i));
			int val_ascii = static_cast<int>(val_orig);
			fprintf(excel_image, "%d \t", val_ascii);
		}
		fprintf(excel_image, "\n");
	}
	fclose(excel_image);

}

Mat resizeImage(Mat image_original, int n) {

	int rows_o = image_original.rows;
	int cols_o = image_original.cols;

	Mat Resize(rows_o + (n - 1), cols_o + (n - 1), CV_8UC1);
	double value = 0;

	for (int j = 0; j < rows_o; j++) {
		for (int i = 0; i < cols_o; i++) {

			value = static_cast<float>(image_original.at<uchar>(Point(j, i)));
			Resize.at<uchar>(Point(j + ((n - 1) / 2), i + ((n - 1) / 2))) = uchar(value);

		}
	}

	return Resize;
}

Mat grayImage(Mat original_image) {

	Mat aux(original_image.cols, original_image.rows, CV_8UC1);

	double blue, green, red, value;
	for (int i = 0; i < original_image.rows; i++) {
		for (int j = 0; j < original_image.cols; j++) {
			blue = original_image.at<Vec3b>(Point(j, i)).val[0];
			green = original_image.at<Vec3b>(Point(j, i)).val[1];
			red = original_image.at<Vec3b>(Point(j, i)).val[2];
			value = blue * 0.114 + green * 0.587 + red * 0.299; //MOIDIFICAR ESTA PARTE 
			aux.at<uchar>(Point(j, i)) = uchar(value);
		}
	}

	return aux;

}

void createKernel(Mat kernel, int n, double sigma) {

	int dim = n / 2;
	int i = 0, j = 0;
	for (int k = -dim; k <= dim; k++) {
		j = 0;
		for (int l = -dim; l <= dim; l++) {

			kernel.at<float>(Point(i, j)) = ((1) / (2 * 3.1416 * sigma * sigma)) * exp(-(pow(k, 2) + pow(l, 2)) / (2 * sigma * sigma));
			j++;
		}
		i++;
	}
}

//Kernels para Sobel
void Gx(Mat gx) {  //x y

	gx.at<float>(Point(0, 0)) = -1;
	gx.at<float>(Point(2, 0)) = 1;
	gx.at<float>(Point(0, 1)) = -2;
	gx.at<float>(Point(2, 1)) = 2;
	gx.at<float>(Point(0, 2)) = -1;
	gx.at<float>(Point(2, 2)) = 1;

}

void Gy(Mat gy) {

	gy.at<float>(Point(0, 0)) = 1;
	gy.at<float>(Point(1, 0)) = 2;
	gy.at<float>(Point(2, 0)) = 1;
	gy.at<float>(Point(0, 2)) = -1;
	gy.at<float>(Point(1, 2)) = -2;
	gy.at<float>(Point(2, 2)) = -1;

}

Mat findDerivates(Mat resize, Mat kernel, int size_kernel) {

	//int dim = (size_kernel - 1) / 2;
	int cols = resize.cols - (size_kernel - 1);
	int rows = resize.cols - (size_kernel - 1);

	//Crear imagen
	Mat main_image = Mat::zeros(rows, cols, CV_8UC1);

	//hacer un recorrido
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double n_value = applyFilterSobel(resize, kernel, size_kernel, j, i);
			main_image.at<uchar>(Point(j, i)) = uchar(n_value);
		}
	}
	//

	return main_image;
}


Mat joinG(Mat gx, Mat gy) {

	int rows = gx.rows;
	int cols = gx.cols;

	double v_gx = 0;
	double v_gy = 0;
	double v_g = 0;

	Mat aux = Mat::zeros(rows, cols, CV_8UC1);


	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			v_gx = static_cast<float>(gx.at<uchar>(Point(j, i)));
			v_gy = static_cast<float>(gy.at<uchar>(Point(j, i)));
			v_g = sqrt((v_gx * v_gx) + (v_gy * v_gy));
			aux.at<uchar>(Point(j, i)) = uchar(v_g);
		}
	}

	return aux;
}


Mat calcularAngulo(Mat gx, Mat gy) {

	int rows = gx.rows;
	int cols = gx.cols;

	double v_gx = 0;
	double v_gy = 0;
	double v_g = 0;

	Mat aux = Mat::zeros(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			v_gx = static_cast<float>(gx.at<uchar>(Point(j, i)));
			v_gy = static_cast<float>(gy.at<uchar>(Point(j, i)));

			v_g = (atan(abs(v_gy) / abs(v_gx))) * (180 / M_PI);
			aux.at<uchar>(Point(j, i)) = uchar(v_g);
		}
	}

	return aux;

}

int valueDegree(double value) {

	int aux = 0;

	double value2 = value; //Arreglar conversion

	if ((0 <= value2 < 22.5) || (157.5 <= value2 <= 180)) {
		aux = 0;
	}

	else if (22.5 <= value2 < 67.5) {
		aux = 45;
	}

	else if (67.5 <= value2 < 157.5) {
		aux = 90;
	}

	else if (112.5 <= value2 < 157.5) {
		aux = 135;
	}

	return aux;
}

Mat NonMaximunSuppresion(Mat image, Mat angulos) {

	int rows = image.rows;
	int cols = image.cols;


	double angulo;
	int categoria = 0;

	double aux1 = 255;
	double aux2 = 255;
	double value_p;

	//obtener valores 

	Mat aux = Mat::zeros(rows, cols, CV_8UC1);

	for (int i = 1; i < rows - 1; i++) {//y
		for (int j = 1; j < cols - 1; j++) {//x

			value_p = static_cast<float>(image.at<uchar>(Point(j, i)));
			angulo = static_cast<float>(angulos.at<uchar>(Point(j, i)));
			categoria = valueDegree(angulo);

			if (categoria == 0) {
				aux1 = static_cast<float>(image.at<uchar>(Point(j + 1, i)));
				aux2 = static_cast<float>(image.at<uchar>(Point(j - 1, i)));
			}
			else if (categoria == 45) {
				aux1 = static_cast<float>(image.at<uchar>(Point(j + 1, i - 1)));
				aux2 = static_cast<float>(image.at<uchar>(Point(j - 1, i + 1)));
			}
			else if (categoria == 90) {
				aux1 = static_cast<float>(image.at<uchar>(Point(j, i + 1)));
				aux2 = static_cast<float>(image.at<uchar>(Point(j, i - 1)));
			}

			else if (categoria == 135) {
				aux1 = static_cast<float>(image.at<uchar>(Point(j + 1, i + 1)));
				aux2 = static_cast<float>(image.at<uchar>(Point(j - 1, i - 1)));
			}

			//Evaluar si son mas grandes que los demas

			if (aux1 > value_p || aux2 > value_p) {
				aux.at<uchar>(Point(j, i)) = uchar(0);
			}
			else {
				double valorn;
				valorn = static_cast<float>(image.at<uchar>(Point(j, i)));
				aux.at<uchar>(Point(j, i)) = uchar(valorn);
			}
		}
	}


	return aux;
}


Mat Umbralizado(Mat image) {

	int rows = image.rows;
	int cols = image.cols;

	double min = 28;
	double max = 118;

	double value = 0;

	Mat aux = Mat::zeros(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			value = static_cast<float>(image.at<uchar>(Point(j, i)));
			//Comparar
			if (value < min) {
				aux.at<uchar>(Point(j, i)) = uchar(0);
				//cout << "hola1";
			}
			else if (min <= value && value <= max) {
				aux.at<uchar>(Point(j, i)) = uchar(25);
				//cout << "hola2";
			}
			else if (value >= max) {
				aux.at<uchar>(Point(j, i)) = static_cast<uchar>(255);
				//cout << "hola3";
			}
		}
	}

	return aux;

}


Mat Histeresis(Mat image) {

	int rows = image.rows;
	int cols = image.cols;

	double value;
	float min = 25;
	double max = 255;


	Mat aux = Mat::zeros(rows, cols, CV_8UC1);

	for (int i = 1; i < rows - 1; i++) {//y
		for (int j = 1; j < cols - 1; j++) {//x
			value = static_cast<float>(image.at<uchar>(Point(j, i)));

			if (value == min) {
				if (static_cast<float>(image.at<uchar>(Point(j - 1, i - 1))) == max || static_cast<float>(image.at<uchar>(Point(j, i - 1))) == max ||
					static_cast<float>(image.at<uchar>(Point(j + 1, i - 1))) == max || static_cast<float>(image.at<uchar>(Point(j - 1, i))) == max ||
					static_cast<float>(image.at<uchar>(Point(j + 1, i))) == max || static_cast<float>(image.at<uchar>(Point(j - 1, i + 1))) == max ||
					static_cast<float>(image.at<uchar>(Point(j, i + 1))) == max || static_cast<float>(image.at<uchar>(Point(j + 1, i + 1))) == max
					) {
					aux.at<uchar>(Point(j, i)) = 255;
				}
				else {
					aux.at<uchar>(Point(j, i)) = uchar(min);
				}

			}

			else {
				aux.at<uchar>(Point(j, i)) = uchar(value);
			}
		}
	}

	return aux;
}


int main()
{
	int n = 0;
	double sigma = 0;

	//Cargar imagen original
	Mat image;
	char imageName[] = "lena.jpg";
	image = imread(imageName);
	int rows = image.rows;
	int cols = image.cols;

	//
	cout << "Introduzca las dimesniones de la matriz n*n" << endl;
	cin >> n;
	cout << "Introduzca el valor de sigma" << endl;
	cin >> sigma;

	if (n % 2 != 1) {
		cerr << "La matriz tiene que ser de número impares: " << endl;
		exit(0);
	}

	imshow("Imagen Original", image);

	//Convertir la imagen original a gris
	Mat gray_image = Mat::zeros(rows, cols, CV_8UC1);
	gray_image = grayImage(image);
	imshow("Imagen en gris", gray_image);


	//Filtro gaussiano
	//Crear imagen con borden
	Mat resize = Mat::zeros(rows + ((n - 1)), cols + (n - 1), CV_8UC1);

	resize = resizeImage(gray_image, n);
	//imshow("BorderImage.png", resize);
	//imwrite("NewImage.png", resize);
	//createExcel(resize);

	//Pasar filtro gaussiano
	Mat gauss_image(rows, cols, CV_8UC1);
	Mat kernel_1 = Mat::zeros(n, n, CV_32F);
	createKernel(kernel_1, n, sigma);
	cout << kernel_1 << endl;

	gauss_image = gaussFilter(resize, kernel_1, n);
	imshow("Gauss_Filter", gauss_image);

	//Ecualización
	Mat equ = Mat::zeros(rows, cols, CV_8UC1);
	equalizeHist(gauss_image, equ);
	imshow("Imagen ecualizada", equ);



	// Sobel
	//Crear Kernels
	Mat gx = Mat::zeros(3, 3, CV_32F);
	Gx(gx);
	//cout << gx << endl;
	Mat gy = Mat::zeros(3, 3, CV_32F);
	Gy(gy);
	//cout << gy << endl;
	//Crear imagen redimensionada
	Mat resize2 = Mat::zeros(gauss_image.rows + 2, gauss_image.cols + 2, CV_8UC1);
	resize2 = resizeImage(equ, 3);
	//Aplicar filtro
	Mat image_gx = Mat::zeros(rows, cols, CV_8UC1);
	image_gx = findDerivates(resize2, gx, 3);
	//imshow("join2", image_gx);
	Mat image_gy = Mat::zeros(rows, cols, CV_8UC1);
	image_gy = findDerivates(resize2, gy, 3);
	//imshow("join1", image_gy);
	//Unir dos gradiente
	Mat G = Mat::zeros(rows, cols, CV_8UC1);
	G = joinG(image_gx, image_gy);
	imshow("G", G);
	//Crear matriz con los angulos
	Mat theta = Mat::zeros(rows, cols, CV_8UC1);
	theta = calcularAngulo(image_gx, image_gy);
	//imshow("Anuglo", theta);



	//Supresión no máxima
	Mat NMS = Mat::zeros(rows, cols, CV_8UC1);
	NMS = NonMaximunSuppresion(G, theta);
	//imshow("NMS", NMS);

	//Umbralizado
	Mat Umb = Mat::zeros(rows, cols, CV_8UC1);
	Umb = Umbralizado(NMS);
	//imshow("Umbralizado", Umb);

	//Histeresis
	Mat His = Mat::zeros(rows, cols, CV_8UC1);
	His = Histeresis(Umb);
	imshow("Histeresis", His);

	//Imprime tamaño de las imagenes
	cout << "Tamaños de las imagenes:" << endl;

	cout << "Original: \t" << image.rows << "x" << image.cols << endl;
	cout << "Imagen a gris: \t" << gray_image.rows << "x" << gray_image.cols << endl;
	cout << "Imagen con Bordes: " << resize.rows << "x" << resize.cols << endl;
	cout << "Filtro Gaus: \t" << gauss_image.rows << "x" << gauss_image.cols << endl;
	cout << "Imagen ecualizada \t" << equ.rows << "x" << equ.cols << endl;
	cout << "Gx: \t" << image_gx.rows << "x" << image_gx.cols << endl;
	cout << "Gy: \t" << image_gy.rows << "x" << image_gy.cols << endl;
	cout << "Magnitud |G|: \t" << G.rows << "x" << G.cols << endl;
	cout << "Supresión No Máxima: \t" << NMS.rows << "x" << NMS.cols << endl;
	cout << "Umbralizad: \t" << Umb.rows << "x" << Umb.cols << endl;
	cout << "Bordes Canny: \t" << His.rows << "x" << His.cols << endl;


	waitKey(0);
	return 1;
}