#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <opencv2/video/background_segm.hpp>


using namespace cv;
using namespace std;

/*odczyt i zapis*/
fstream zapis;
string sciezkaZapisu = "./wyniki/Tomczak_Filip.txt";
vector<string>sekwencje;
string sciezkaSekwencja;
string sekwencja;
fstream odczyt;
string sciezkaNazwSekwencji = "./nazwy_sekwencji/nazwy_sekwencji.txt";
/**/
VideoCapture video;
Mat frameOrginal, framePreprocess;
Ptr<BackgroundSubtractorMOG2> backgroundSubtraction = createBackgroundSubtractorMOG2();
Mat structurElement5 = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat structurElement3 = getStructuringElement(MORPH_RECT, Size(3, 3));
vector<vector<Point>>kontury;
/*car/truck right*/
Rect rightArea = Rect(290, 145, 30, 120);
Rect rightArea2 = Rect(250, 145, 30, 120);
/*car/truck left*/
Rect leftArea = Rect(130, 50, 20, 90);
Rect leftArea2 = Rect(160, 50, 20, 90);
/*people left*/
Rect peopleAreaLeft2 = Rect(40, 320, 10, 120);
Rect peopleAreaLeft = Rect(53, 320, 10, 120);
Rect peopleAreaLeftCount = Rect(66, 320, 10, 120);
/*people right*/
Rect peopleAreaRight2 = Rect(655, 320, 10, 120);
Rect peopleAreaRight = Rect(642, 320, 10, 120);
Rect peopleAreaRightCount = Rect(629, 320, 10, 120);
/*bike left*/
Rect bikeAreaLeft2 = Rect(40, 280, 15, 65);
Rect bikeAreaLeft = Rect(60, 280, 15, 65);
Rect bikeAreaLeftCount = Rect(80, 280, 15, 65);
/*bike right*/
Rect bikeAreaRight2 = Rect(650, 280, 15, 65);
Rect bikeAreaRight = Rect(630, 280, 15, 65);
Rect bikeAreaRightCount = Rect(610, 280, 15, 65);
/*tram*/
Rect tramAreaLeft = Rect(150, 80, 50, 80);
Rect tramArea = Rect(220, 80, 50, 80);
Rect tramAreaRight = Rect(290, 80, 50, 80);
/*kolory*/
Scalar Red = Scalar(0, 0, 255);
Scalar Green = Scalar(0, 255, 0);
Scalar Blue = Scalar(255, 0, 0);
Scalar Yellow = Scalar(0, 255, 255);
Scalar White = Scalar(255, 255, 255);
/*zmienne pomocnicze*/
int rightCarCounter = 0;
int leftCarCounter = 0;
int rightTruckCounter = 0;
int leftTruckCounter = 0;
int tramCounter = 0;
int peopleCounter = 0;
int bikeCounter = 0;
/*flagi auta*/
bool flagRightCar = false;
bool flagLeftCar = false;
/*flagi ludzie*/
bool flagRightPeople = false;
bool flagLeftPeople = false;
/*flagi rower*/
bool flagRightBike = false;
bool flagLeftBike = false;
/*flaga tramwaj*/
bool flagTram = false;
/*funkcje*/
bool crossRightArea(Point srodek);
bool crossLeftArea(Point srodek);
bool crossPeopleArea(Point srodek);
bool crossTramArea(Point srodek);

void countObject(Rect poleKontur, Point srodek);

void whatObject(Rect poleKontur, Point srodek);

void refreshData();

void preprocessOperation(Mat& frameOrginal);

float fulfillFactor(Rect prostokat);

bool carOrCars(Rect prostokat);

void drawRectangles(Scalar Color, Scalar Color2);

void drawRectangle(Scalar Color, Rect rectangleArea, int contourSize);

void clearNoise(Mat& frame);

void writeData();


int main()
{
	zapis.open(sciezkaZapisu, ios::out | ios::trunc);
	zapis.close();

	odczyt.open(sciezkaNazwSekwencji, ios::in);

	if (!odczyt.is_open())
		cout << "blad otwarcia pliku!" << endl;
	else
		cout << "otwarto plik!" << endl;
	while (getline(odczyt, sekwencja))
		sekwencje.push_back(sekwencja);
	odczyt.close();

	for (int i = 0; i < sekwencje.size(); i++) {
		sciezkaSekwencja = "./sekwencje_wideo/" + sekwencje[i];
		cout << sekwencje[i] << endl;
		VideoCapture video(sciezkaSekwencja);
		while (1)
		{
			video >> frameOrginal;
			if (frameOrginal.empty())
			{
				writeData();
				break;
			}
			preprocessOperation(frameOrginal);
			drawRectangles(Red, Yellow);

			findContours(framePreprocess, kontury, RETR_TREE, CHAIN_APPROX_SIMPLE); //retr ext 
			if (kontury.size() > 0)
			{
				vector<vector<Point>>wierzcholki(kontury.size());
				for (int i = 0; i < kontury.size(); i++)
					convexHull(kontury[i], wierzcholki[i]);
				vector<Rect>prostokat(wierzcholki.size());
				vector<vector<Point>>konturyProstokata(wierzcholki.size());
				for (int i = 0; i < wierzcholki.size(); i++)
				{
					approxPolyDP(wierzcholki[i], konturyProstokata[i], 1, true); //wygladza wierzcholki
					prostokat[i] = boundingRect(konturyProstokata[i]);
					if (prostokat[i].area() > 1500) {
						Point srodek = Point(prostokat[i].x + prostokat[i].width / 2, prostokat[i].y + prostokat[i].height / 2);
						circle(frameOrginal, srodek, 2, Scalar(255, 255, 255), -1);
						whatObject(prostokat[i], srodek);
						countObject(prostokat[i], srodek);
					}
				}
				wierzcholki.clear();
				kontury.clear();
				konturyProstokata.clear();
			}
			refreshData();
			//imshow("licznik aut", frameOrginal);
			/*
			if (waitKey(0) == 49) {
				continue;
			}
			*/
			if (waitKey(30) == 27) break;
		}
	}
	//system("pause");
	return 0;
}

void writeData()
{
	zapis.open(sciezkaZapisu, ios::out | ios::app);
	zapis << leftCarCounter << ", " << rightCarCounter << ", " << leftTruckCounter << ", " << rightTruckCounter << ", " << tramCounter << ", " << peopleCounter << ", " << bikeCounter << endl;
	zapis.close();
	rightCarCounter = 0;
	leftCarCounter = 0;
	rightTruckCounter = 0;
	leftTruckCounter = 0;
	tramCounter = 0;
	peopleCounter = 0;
	bikeCounter = 0;
}

void preprocessOperation(Mat& frameOrginal)
{
	resize(frameOrginal, frameOrginal, Size(700, 450));
	cvtColor(frameOrginal, framePreprocess, COLOR_BGR2GRAY);
	GaussianBlur(framePreprocess, framePreprocess, Size(13, 13), 0);
	backgroundSubtraction->apply(framePreprocess, framePreprocess);
	medianBlur(framePreprocess, framePreprocess, 3);

	erode(framePreprocess, framePreprocess, structurElement3);
	dilate(framePreprocess, framePreprocess, structurElement5, Point(-1, -1), 2);

	clearNoise(framePreprocess);
	threshold(framePreprocess, framePreprocess, 100, 255, THRESH_BINARY);
	//imshow("preprocess", framePreprocess);
}

bool carOrCars(Rect prostokatProgowanie)
{
	rectangle(framePreprocess, prostokatProgowanie, Scalar(255, 255, 255), 1);
	int licznikBialychPixeli = 0;
	int licznikCzarnychPixeli = 0;
	int sumaBialych1 = 0;
	int sumaCzarnych1 = 0;
	int sumaBialych2 = 0;
	int sumaCzarnych2 = 0;
	int sumaBialych3 = 0;
	int sumaCzarnych3 = 0;
	int sumaBialych4 = 0;
	int sumaCzarnych4 = 0;
	// Icwiartka
	for (int j = prostokatProgowanie.y; j < (prostokatProgowanie.y + prostokatProgowanie.height / 2); j++) { //sprawdzanie po kolumnach 
		for (int k = prostokatProgowanie.x; k < (prostokatProgowanie.x + prostokatProgowanie.width / 2); k++) {
			if (framePreprocess.at<uchar>(j, k) > 1)
				licznikBialychPixeli++;
			else
				licznikCzarnychPixeli++;
		}
		if (licznikBialychPixeli > licznikCzarnychPixeli)
			sumaBialych1++;
		else
			sumaCzarnych1++;
		licznikBialychPixeli = 0;
		licznikCzarnychPixeli = 0;
	}
	// IIcwiartka
	for (int j = prostokatProgowanie.y; j < (prostokatProgowanie.y + prostokatProgowanie.height / 2); j++) {
		for (int k = prostokatProgowanie.x + prostokatProgowanie.width / 2; k < (prostokatProgowanie.x + prostokatProgowanie.width); k++) {
			if (framePreprocess.at<uchar>(j, k) > 1)
				licznikBialychPixeli++;
			else
				licznikCzarnychPixeli++;
		}
		if (licznikBialychPixeli > licznikCzarnychPixeli)
			sumaBialych2++;
		else
			sumaCzarnych2++;
		licznikBialychPixeli = 0;
		licznikCzarnychPixeli = 0;
	}
	// IIIcwiartka
	for (int j = prostokatProgowanie.y + prostokatProgowanie.height / 2; j < (prostokatProgowanie.y + prostokatProgowanie.height); j++) {
		for (int k = prostokatProgowanie.x; k < (prostokatProgowanie.x + prostokatProgowanie.width / 2); k++) {
			if (framePreprocess.at<uchar>(j, k) > 1)
				licznikBialychPixeli++;
			else
				licznikCzarnychPixeli++;
		}
		if (licznikBialychPixeli > licznikCzarnychPixeli)
			sumaBialych3++;
		else
			sumaCzarnych3++;
		licznikBialychPixeli = 0;
		licznikCzarnychPixeli = 0;
	}
	// IVcwiartka
	for (int j = prostokatProgowanie.y + prostokatProgowanie.height / 2; j < (prostokatProgowanie.y + prostokatProgowanie.height); j++) {
		for (int k = prostokatProgowanie.x + prostokatProgowanie.width / 2; k < (prostokatProgowanie.x + prostokatProgowanie.width); k++) {
			if (framePreprocess.at<uchar>(j, k) > 1)
				licznikBialychPixeli++;
			else
				licznikCzarnychPixeli++;
		}
		if (licznikBialychPixeli > licznikCzarnychPixeli)
			sumaBialych4++;
		else
			sumaCzarnych4++;
		licznikBialychPixeli = 0;
		licznikCzarnychPixeli = 0;
	}
	/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ZMIANA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
	if (((1.2*(sumaCzarnych1 + sumaCzarnych4) > (sumaBialych1 + sumaBialych4)) || (1.2*(sumaCzarnych2 + sumaCzarnych3) > (sumaBialych2 + sumaBialych3))) && fulfillFactor(prostokatProgowanie) < 0.75) {
		return true;
	}
	return false;
}

bool crossLeftArea(Point srodek)
{
	if (flagLeftCar == false)
		if (leftArea2.contains(srodek))
		{
			drawRectangle(Green, leftArea2, 2);
			flagLeftCar = true;
		}
	if (flagLeftCar == true)
		if (leftArea.contains(srodek))
		{
			drawRectangle(Green, leftArea, 2);
			flagLeftCar = false;
			return true;
		}
	return false;
}

bool crossRightArea(Point srodek)
{
	if (flagRightCar == false)
		if (rightArea2.contains(srodek))
		{
			drawRectangle(Green, rightArea2, 2);
			flagRightCar = true;
		}
	if (flagRightCar == true)
		if (rightArea.contains(srodek))
		{
			drawRectangle(Green, rightArea, 2);
			flagRightCar = false;
			return true;
		}
	return false;
}

bool crossPeopleArea(Point srodek)
{
	/*lewa strone*/
	if (peopleAreaLeft2.contains(srodek)) {
		drawRectangle(Green, peopleAreaLeft2, 2);
		flagLeftPeople = false;
	}

	if (flagLeftPeople == false && peopleAreaLeft.contains(srodek)) {
		drawRectangle(Green, peopleAreaLeft, 2);
		flagLeftPeople = true;
	}

	if (flagLeftPeople == true && peopleAreaLeftCount.contains(srodek)) {
		drawRectangle(Green, peopleAreaLeftCount, 2);
		flagLeftPeople = false;
		flagRightPeople = false;
		return true;
	}
	/*prawa strona*/
	if (peopleAreaRight2.contains(srodek)) {
		drawRectangle(Green, peopleAreaRight2, 2);
		flagRightPeople = false;
	}

	if (flagRightPeople == false && peopleAreaRight.contains(srodek)) {
		drawRectangle(Green, peopleAreaRight, 2);
		flagRightPeople = true;
	}

	if (flagRightPeople == true && peopleAreaRightCount.contains(srodek)) {
		drawRectangle(Green, peopleAreaRightCount, 2);
		flagRightPeople = false;
		flagLeftPeople = false;
		return true;
	}

	return false;
}

bool crossBikeArea(Point srodek)
{
	/*lewa strone*/
	if (bikeAreaLeft2.contains(srodek)) {
		drawRectangle(Green, bikeAreaLeft2, 2);
		flagLeftBike = false;
	}

	if (flagLeftBike == false && bikeAreaLeft.contains(srodek)) {
		drawRectangle(Green, bikeAreaLeft, 2);
		flagLeftBike = true;
	}

	if (flagLeftBike == true && bikeAreaLeftCount.contains(srodek)) {
		drawRectangle(Green, bikeAreaLeftCount, 2);
		flagLeftBike = false;
		flagRightBike = false;
		return true;
	}
	/*prawa strona*/
	if (bikeAreaRight2.contains(srodek)) {
		drawRectangle(Green, bikeAreaRight2, 2);
		flagRightBike = false;
	}

	if (flagRightBike == false && bikeAreaRight.contains(srodek)) {
		drawRectangle(Green, bikeAreaRight, 2);
		flagRightBike = true;
	}

	if (flagRightBike == true && bikeAreaRightCount.contains(srodek)) {
		drawRectangle(Green, bikeAreaRightCount, 2);
		flagRightBike = false;
		flagLeftBike = false;
		return true;
	}

	return false;
}

bool crossTramArea(Point srodek)
{
	if (tramAreaRight.contains(srodek) && flagTram == false)
	{
		drawRectangle(Green, tramAreaRight, 2);
		flagTram = true;
	}
	if (tramAreaLeft.contains(srodek) && flagTram == false)
	{
		drawRectangle(Green, tramAreaLeft, 2);
		flagTram = true;
	}

	if (tramArea.contains(srodek) && flagTram == true)
	{
		drawRectangle(Green, tramArea, 2);
		flagTram = false;
		return true;
	}
	return false;
}

void countObject(Rect poleKontur, Point srodek)
{
	/*pieszy*/
	if (poleKontur.height > poleKontur.width*1.3 || (poleKontur.height*1.1>poleKontur.width && poleKontur.height*0.9<poleKontur.width)) { //osoba lub osoba z wozkiem
		if (crossPeopleArea(srodek)) {
			peopleCounter++;
		}
	}
	/*rower*/
	if (poleKontur.height > poleKontur.width && poleKontur.width*1.3>poleKontur.height) {
		if (crossBikeArea(srodek)) {
			bikeCounter++;
		}
	}
	/*auto w prawo*/
	if (poleKontur.area() > 2000 && poleKontur.area() < 30000 && poleKontur.height < 1.2*poleKontur.width && 3 * poleKontur.height > poleKontur.width) {
		if (crossRightArea(srodek))
		{
			/*************************/
			fulfillFactor(poleKontur);
			/*************************/
			rightCarCounter++;
			if (carOrCars(poleKontur))
				rightCarCounter++;
		}
	}
	/*auto w lewo*/ /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TRZEBA POKOMBINOWAC Z WYSOKOSCIA, ZEBY ODRZUCALO SLAD PO TRAMWAJU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
	if (poleKontur.area() > 2000 && poleKontur.area() < 20000 && poleKontur.height < 1.2*poleKontur.width && 3 * poleKontur.height > poleKontur.width && poleKontur.height<100) {
		if (crossLeftArea(srodek))
		{
			/*************************/
			fulfillFactor(poleKontur);
			/*************************/
			leftCarCounter++;
			if (carOrCars(poleKontur))
				leftCarCounter++;
		}
	}
	/*tir w prawo*/
	if (poleKontur.area() > 30000 && poleKontur.area() < 90000 && (poleKontur.width > poleKontur.height)) {
		if (crossRightArea(srodek)) {
			/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ZMIANA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
			if (carOrCars(poleKontur)) {
				rightCarCounter++;
				rightCarCounter++;
			}
			else
				rightTruckCounter++;
		}
	}
	/*tir w lewo*/
	if (poleKontur.area() > 20000 && poleKontur.area() < 40000 && (poleKontur.width > poleKontur.height)) {
		if (crossLeftArea(srodek)) {
			leftTruckCounter++;
		}
	}
	/*tramwaj*/
	if (poleKontur.area() > 50000) {
		if (crossTramArea(srodek)) {
			tramCounter++;
		}
	}
}

void whatObject(Rect poleKontur, Point srodek)
{
	string movingObject = "";

	if (poleKontur.height > poleKontur.width) {
		drawRectangle(Blue, poleKontur, 2);
		putText(frameOrginal, "pieszy", Point(poleKontur.x + poleKontur.width / 5, poleKontur.y + poleKontur.height / 2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);
		movingObject = "pieszy";
	}
	if (poleKontur.area() > 2000 && poleKontur.area() < 30000 && poleKontur.height < 1.2*poleKontur.width && 3 * poleKontur.height > poleKontur.width) {
		drawRectangle(Blue, poleKontur, 2);
		putText(frameOrginal, "auto", Point(poleKontur.x + poleKontur.width / 5, poleKontur.y + poleKontur.height / 2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);
		movingObject = "auto";
	}
	if (poleKontur.area() > 30000 && poleKontur.area() < 50000 && (poleKontur.width>poleKontur.height)) {
		drawRectangle(Blue, poleKontur, 2);
		putText(frameOrginal, "tir", Point(poleKontur.x + poleKontur.width / 5, poleKontur.y + poleKontur.height / 2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);
		movingObject = "tir";
	}
	if (poleKontur.area() > 50000) {
		drawRectangle(Blue, poleKontur, 2);
		putText(frameOrginal, "tramwaj", Point(poleKontur.x + poleKontur.width / 5, poleKontur.y + poleKontur.height / 2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);
		movingObject = "tramwaj";
	}
}

float fulfillFactor(Rect prostokatProgowanie)
{
	float wspolczynnik = 0;

	if (prostokatProgowanie.area() > 2500) //od tej wartosci wykrywane sa auta
	{
		rectangle(framePreprocess, prostokatProgowanie, Scalar(255, 255, 255), 1);
		int licznikBialychPixeli = 0;
		for (int j = prostokatProgowanie.y; j < (prostokatProgowanie.y + prostokatProgowanie.height); j++)
			for (int k = prostokatProgowanie.x; k < (prostokatProgowanie.x + prostokatProgowanie.width); k++)
				if (framePreprocess.at<uchar>(j, k) > 10)
					licznikBialychPixeli++;
		wspolczynnik = (float)licznikBialychPixeli / prostokatProgowanie.area();
	}
	return wspolczynnik;
}

void clearNoise(Mat& frame)
{
	int licznikSzumu1 = 0;
	float wspolczynnikSzumu1 = 0;
	Rect pavementNoise = Rect(10, 400, 680, 45);
	for (int j = pavementNoise.y; j < pavementNoise.y + pavementNoise.height; j++)
		for (int i = pavementNoise.x; i < pavementNoise.x + pavementNoise.width; i++)
			if (frame.at<uchar>(j, i) > 0)
				licznikSzumu1++;
	wspolczynnikSzumu1 = (float)licznikSzumu1 / pavementNoise.area();

	int licznikSzumu2 = 0;
	float wspolczynnikSzumu2 = 0;
	Rect upNoise = Rect(1, 1, 50, 20);
	for (int j = upNoise.y; j < upNoise.y + upNoise.height; j++)
		for (int i = upNoise.x; i < upNoise.x + upNoise.width; i++)
			if (frame.at<uchar>(j, i) > 0)
				licznikSzumu2++;
	wspolczynnikSzumu2 = (float)licznikSzumu2 / upNoise.area();

	if (wspolczynnikSzumu1 > 0.05 && wspolczynnikSzumu2 > 0.05) //posprawdzac i pokombinowac
	{
		threshold(frame, frame, 250, 255, THRESH_BINARY);
	}
}

void refreshData()
{
	string counterPeople = format("Piesi: %d", peopleCounter);
	string counterBike = format("Rower: %d", bikeCounter);
	string counterCarRight = format("Auta: %d", rightCarCounter);
	string counterCarLeft = format("Auta: %d", leftCarCounter);
	string counterTruckRight = format("Tiry: %d", rightTruckCounter);
	string counterTruckLeft = format("Tiry: %d", leftTruckCounter);
	string counterTram = format("Tramwaje: %d", tramCounter);

	putText(frameOrginal, counterCarLeft, Point(10, 110), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);
	putText(frameOrginal, counterTruckLeft, Point(10, 130), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);

	putText(frameOrginal, counterCarRight, Point(10, 210), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);
	putText(frameOrginal, counterTruckRight, Point(10, 230), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);

	putText(frameOrginal, counterBike, Point(210, 350), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);
	putText(frameOrginal, counterPeople, Point(210, 370), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);
	putText(frameOrginal, counterTram, Point(210, 390), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 250, 0), 1, 8);
}

void drawRectangle(Scalar Color, Rect rectangleArea, int contourSize)
{
	rectangle(frameOrginal, rectangleArea, Color, contourSize, 8, 0);
}

void drawRectangles(Scalar Color, Scalar Color2)
{
	rectangle(frameOrginal, rightArea2, Color, 1, 8, 0);
	rectangle(frameOrginal, rightArea, Color, 1, 8, 0);

	rectangle(frameOrginal, leftArea2, Color, 1, 8, 0);
	rectangle(frameOrginal, leftArea, Color, 1, 8, 0);

	rectangle(frameOrginal, peopleAreaLeftCount, Color, 1, 8, 0);
	rectangle(frameOrginal, peopleAreaRightCount, Color, 1, 8, 0);
	rectangle(frameOrginal, peopleAreaLeft, Color, 1, 8, 0);
	rectangle(frameOrginal, peopleAreaRight, Color, 1, 8, 0);
	rectangle(frameOrginal, peopleAreaLeft2, Color, 1, 8, 0);
	rectangle(frameOrginal, peopleAreaRight2, Color, 1, 8, 0);

	rectangle(frameOrginal, bikeAreaLeftCount, Color2, 1, 8, 0);
	rectangle(frameOrginal, bikeAreaRightCount, Color2, 1, 8, 0);
	rectangle(frameOrginal, bikeAreaLeft, Color2, 1, 8, 0);
	rectangle(frameOrginal, bikeAreaRight, Color2, 1, 8, 0);
	rectangle(frameOrginal, bikeAreaLeft2, Color2, 1, 8, 0);
	rectangle(frameOrginal, bikeAreaRight2, Color2, 1, 8, 0);

	rectangle(frameOrginal, tramAreaLeft, Color2, 1, 8, 0);
	rectangle(frameOrginal, tramArea, Color2, 1, 8, 0);
	rectangle(frameOrginal, tramAreaRight, Color2, 1, 8, 0);
}