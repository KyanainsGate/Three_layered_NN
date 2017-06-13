#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>
#include<process.h>
//#pragma warning(disable:4996) //freopenに関するエラーを完全に無視する模様

/*----終了条件-----------------------------------------------*/
#define times 6000		//学習回数の上限
#define E_alloable 0.04	//1パターンごとの誤差の総和
#define NumOfSampling 100 //取得データ数
/*----出力先の指定と内容------------------------------------*/
FILE *fpFofChange;		//結合荷重とバイアスの変化を記録
FILE *fpForValue;		//学習終了後の全パラメータを記録
						/*-----諸変数の定義----------------------------------------*/
#define IN 2					//入力ユニット数
#define HN 2					//中間層ユニット数
#define ON 1					//出力層ユニット数
#define LAY 1					//隠れ層の総数
double OT_IN[IN];				//入力層の出力[IN]
double OT_HD[HN];			//中間層の出力[HN]
double OT_OT[ON];				//出力層の出力[ON]
double W_IN_HD[HN][IN];			//入力層から隠れ層の結合係数W_ji([HN][IN]
double CW_HD[HN];			//中間層ユニットのオフセット[HN]
double W_HD_OT[ON][HN];			//中間層から出力層の結合係数V_kj([ON][HN])
double CW_OT[ON];				//出力層ユニットのオフセット[ON]
double TEACH[ON];				//教師信号(structから一時的に1or0を格納)[ON]
double DEL_OT[ON];				//誤差（出力層)[ON]
double DEL_HD[HN];				//誤差（隠れ層)[HN]
double alpha;					//結合荷重の変化量
double beta;					//オフセットの変化量

								/*-----解くべき問題条件:EXOR------------------------------*/

#define SAMPLINGNUM 4	//教師モデルの数（EXORなので4つ）
struct {
	int input[2];	//入力データ
	int tch[1];		//教師データ
} indata[SAMPLINGNUM] = { { 1, 0, 1 },{ 0, 1, 1 },{ 0, 0, 0 },{ 1, 1, 0 } };	  //構造体による教師入力,本データはEXOR


																				  /*-----シグモイド関数-------------------------------------*/
#define u0 1			//constant for sigmoid(Temperature)
double ru0 = 2 / u0;	//今後定数になるので決め解いた値
double sigmf(double x) {
	return 1 / (1.0 + exp(-1 * u0 * x *0.5));
}

/*-----パラメータ推移解析用関数（新）-----------------------*/

void SamplingValueChangeNew(int num, double totalError) {
	int S = times / NumOfSampling;
	int m, k;
	char filename[256];
	if (num == 0) {
		/*Making File Name*/
		sprintf_s(filename, 256, "HN%dTime%dHL%d.csv", HN, times, LAY);
		/*FileOpen*/
		fopen_s(&fpFofChange, filename, "w");
	}
	/*Produce Contents*/
	if (num % S == 0) {
		if (num == 0) {
			fprintf(fpFofChange, "LoopTime,");
			for (k = 0; k < HN; k++) {
				for (m = 0; m < IN; m++) {
					fprintf(fpFofChange, "W_IN_HD[%d][%d],", k, m);
				}
				fprintf(fpFofChange, "CW_HD_[%d],", k);
			}
			for (k = 0; k < ON; k++) {
				for (m = 0; m < HN; m++) {
					fprintf(fpFofChange, "W_HD_OT[%d][%d],", k, m);
				}
				fprintf(fpFofChange, "CW_OT_[%d],", k);
			}
			fprintf(fpFofChange, "FinalError\n");
		}
		//値の出力
		fprintf(fpFofChange, "%d,", num);
		for (k = 0; k < HN; k++) {
			for (m = 0; m < IN; m++) {
				fprintf(fpFofChange, "%f,", W_IN_HD[k][m]);
			}
			fprintf(fpFofChange, "%f,", CW_HD[k]);
		}
		for (k = 0; k < ON; k++) {
			for (m = 0; m < HN; m++) {
				fprintf(fpFofChange, "%f,", W_HD_OT[k][m]);
			}
			fprintf(fpFofChange, "%f,", CW_OT[k]);
		}
		fprintf(fpFofChange, "%f\n", totalError);
	}

	if (num == times) { fclose(fpFofChange); }

	return;
}

/*-----Pythonグラフ出力用関数-------------------------------*/

void forGraphplot(void) {
	int i, j, k;
	char filename[100];
	/*Making File Name*/
	sprintf_s(filename, 100, "plotHN%dTime%dHL%d.csv", HN, times, LAY);
	/*FileOpen*/
	fopen_s(&fpForValue, filename, "w");

	fprintf(fpForValue, ",");
	for (j = 0; j < HN; j++) {
		for (i = 0; i < IN; i++) {
			fprintf(fpForValue, "W%d%d,", j, i);
		}
		fprintf(fpForValue, "th%d,", j);
	}
	for (k = 0; k < ON; k++) {
		for (j = 0; j < HN; j++) {
			fprintf(fpForValue, "V%d%d,", k, j);
		}
		fprintf(fpForValue, "gam%d,", k);
	}
	fprintf(fpForValue, "IN,HN,ON\n");
	//数値計算開始
	fprintf(fpForValue, "x,");
	for (j = 0; j < HN; j++) {
		for (i = 0; i < IN; i++) {
			fprintf(fpForValue, "%f,", W_IN_HD[j][i]);
		}
		fprintf(fpForValue, "%f,", CW_HD[j]);
	}
	for (k = 0; k < ON; k++) {
		for (j = 0; j < HN; j++) {
			fprintf(fpForValue, "%f,", W_HD_OT[k][j]);
		}
		fprintf(fpForValue, "%f,", CW_OT[k]);
	}
	fprintf(fpForValue, "%d,%d,%d\n", IN, HN, ON);
	return;
}

/*-----メイン関数-------------------------------------------*/
int main(void) {

	int k, m, i;
	//int a;

	double inival;/*inivalが加算計算のカウンターとして機能*/

				  /*-------------------------------------------------------------------------------------*/
				  /*結合荷重の初期化:ランダム*/
	double initial_w = 0.0002;
	//入力・中間層区間・・・・W_IN_HD[HN][IN]
	W_IN_HD[0][0] = initial_w;
	W_IN_HD[0][1] = 0.078;
	W_IN_HD[1][0] = -0.423;
	W_IN_HD[1][1] = 0.009;
	//中間・隠れ層区間・・・・W_HD_OT[ON][HN]
	W_HD_OT[0][0] = 0.000908;
	W_HD_OT[0][1] = -0.0087;
	W_HD_OT[0][2] = -0.0048;
	W_HD_OT[0][3] = -0.0087;

	//各閾値
	//隠れ層
	CW_HD[0] = 0.0089;
	CW_HD[1] = -0.098;
	//出力層
	CW_OT[0] = 0.0023;
	//学習係数
	alpha = 0.7;
	beta = 0.7;
	/*-------------------------------------------------------------------------------------*/

	int loop1;		//教師モデル採取用のループ。
	int loop = 0;		//学習ループ数の初期化
	double E_t;		//各パターンの総和誤差

	while (loop < times) {
		double error = 0.0;	//学習回数loop時のerrorを初期化

							/*--------------ある教師モデルに対するループが始動----------------------------------*/
		for (loop1 = 0; loop1 < SAMPLINGNUM; loop1++) {
			//printf("Leaning Patern %d\n", indata[loop1].tch[0]);

			/*順伝播の処理*/
			/*入力層の出力をセットする*/
			for (k = 0; k < IN; k++) {
				OT_IN[k] = (double)indata[loop1].input[k];
				//printf("%f\n", OT_IN[k]);
			}

			/*中間層の出力を求める*/
			for (k = 0; k < HN; k++) {		//H_1,H_2,...を求める
				inival = 0.0;				//繰り返し計算における初期値のセット
				for (m = 0; m < IN; m++) {	//W_ji*I_iの加算が目的
					inival += (W_IN_HD[k][m])*(OT_IN[m]);
				}
				inival += CW_HD[k];			//最後に閾値を加算,今inivalが隠れ層ポテンシャルH_jv
				OT_HD[k] = sigmf(inival);	//写像関数
			}

			/*隠れ層レイヤー間の伝播処理*/
			for (i = 0; i < LAY; i++) {

			}

			/*出力層の出力を求める*/
			for (k = 0; k < ON; k++) {
				inival = 0.0;
				for (m = 0; m < HN; m++) {
					inival += (W_HD_OT[k][m])*(OT_HD[m]);
				}
				inival += CW_OT[k];			//最後に閾値を加算,今inivalが出力層ポテンシャルS_k
				OT_OT[k] = sigmf(inival);	//写像シグモイド関数
			}

			/*逆伝播の処理*/
			/*誤差(出力層)の算出*/
			double wk, wkb;									//wkは出力信号O_kを、wkbはO_kとT_kとの差を算出するのに用いる

			for (m = 0; m < ON; m++) {						//DEL_OT,すなわちHD-OT間の誤差を計算
				TEACH[m] = (double)indata[loop1].tch[m];	//loop１における教師出力を持ってくる
				wk = OT_OT[m];
				wkb = TEACH[m] - OT_OT[m];	//教師信号と出力との差(δ_k = T_k - O_k)
				error += (wkb)*(wkb) / 2;	//出力層mにおける平均二乗誤差,ΣE_mp,m=0から加算
				DEL_OT[m] = wkb * ru0 * wk * (1.0 - wk);
			}
			/*誤差(隠れ層)の算出*/
			for (k = 0; k < HN; k++) {	//DEL_HD,すなわちHD-IN間の誤差を計算
				inival = 0.0;
				for (m = 0; m < ON; m++) {
					inival += (DEL_OT[m] * W_HD_OT[m][k]);
				}
				wk = OT_HD[k];			//先程とは違い、今度はHDの出力信号O_j
				DEL_HD[k] = inival * ru0 * wk * (1.0 - wk);
			}
			/*結合荷重の更新*/
			/*隠れ層-出力層間*/
			for (k = 0; k < ON; k++) {
				for (m = 0; m < HN; m++) {
					W_HD_OT[k][m] += (alpha * DEL_OT[k] * OT_HD[m]);	//結合荷重V_kjの更新
				}
				CW_OT[k] += (beta * DEL_OT[k]);				//offsetの更新
			}
			/*入力層-隠れ層間*/
			for (k = 0; k < HN; k++) {
				for (m = 0; m < IN; m++) {
					W_IN_HD[k][m] += (alpha * DEL_HD[k] * OT_IN[m]);
				}
				CW_HD[k] += (beta * DEL_HD[k]);
			}
		}
		E_t = error;			//E_tに今までのすべてのパターンの誤差を引き渡す

								/*--------------------ある教師モデルに対するループが終了、次の教師モデル採取へ------------------------------*/

								/*-------------------パラメータ推移を観測する出力を行う------------------------------*/
		SamplingValueChangeNew(loop, E_t);
		/*-----------------------------------------------------------------------------------*/

		loop++;//学習回数の更新
	}
	/*-------------------------------------学習の終了-----------------------------------------------------------*/

	/*-------------------収束後のパラメータを吐き出す-----------------------------------*/
	forGraphplot();
	/*----------------------------------------------------------------------------------*/
	return 0;
}
