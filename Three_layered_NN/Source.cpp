#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>
#include<process.h>
//#pragma warning(disable:4996) //freopen�Ɋւ���G���[�����S�ɖ�������͗l

/*----�I������-----------------------------------------------*/
#define times 6000		//�w�K�񐔂̏��
#define E_alloable 0.04	//1�p�^�[�����Ƃ̌덷�̑��a
#define NumOfSampling 100 //�擾�f�[�^��
/*----�o�͐�̎w��Ɠ��e------------------------------------*/
FILE *fpFofChange;		//�����׏d�ƃo�C�A�X�̕ω����L�^
FILE *fpForValue;		//�w�K�I����̑S�p�����[�^���L�^
						/*-----���ϐ��̒�`----------------------------------------*/
#define IN 2					//���̓��j�b�g��
#define HN 2					//���ԑw���j�b�g��
#define ON 1					//�o�͑w���j�b�g��
#define LAY 1					//�B��w�̑���
double OT_IN[IN];				//���͑w�̏o��[IN]
double OT_HD[HN];			//���ԑw�̏o��[HN]
double OT_OT[ON];				//�o�͑w�̏o��[ON]
double W_IN_HD[HN][IN];			//���͑w����B��w�̌����W��W_ji([HN][IN]
double CW_HD[HN];			//���ԑw���j�b�g�̃I�t�Z�b�g[HN]
double W_HD_OT[ON][HN];			//���ԑw����o�͑w�̌����W��V_kj([ON][HN])
double CW_OT[ON];				//�o�͑w���j�b�g�̃I�t�Z�b�g[ON]
double TEACH[ON];				//���t�M��(struct����ꎞ�I��1or0���i�[)[ON]
double DEL_OT[ON];				//�덷�i�o�͑w)[ON]
double DEL_HD[HN];				//�덷�i�B��w)[HN]
double alpha;					//�����׏d�̕ω���
double beta;					//�I�t�Z�b�g�̕ω���

								/*-----�����ׂ�������:EXOR------------------------------*/

#define SAMPLINGNUM 4	//���t���f���̐��iEXOR�Ȃ̂�4�j
struct {
	int input[2];	//���̓f�[�^
	int tch[1];		//���t�f�[�^
} indata[SAMPLINGNUM] = { { 1, 0, 1 },{ 0, 1, 1 },{ 0, 0, 0 },{ 1, 1, 0 } };	  //�\���̂ɂ�鋳�t����,�{�f�[�^��EXOR


																				  /*-----�V�O���C�h�֐�-------------------------------------*/
#define u0 1			//constant for sigmoid(Temperature)
double ru0 = 2 / u0;	//����萔�ɂȂ�̂Ō��߉������l
double sigmf(double x) {
	return 1 / (1.0 + exp(-1 * u0 * x *0.5));
}

/*-----�p�����[�^���ډ�͗p�֐��i�V�j-----------------------*/

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
		//�l�̏o��
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

/*-----Python�O���t�o�͗p�֐�-------------------------------*/

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
	//���l�v�Z�J�n
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

/*-----���C���֐�-------------------------------------------*/
int main(void) {

	int k, m, i;
	//int a;

	double inival;/*inival�����Z�v�Z�̃J�E���^�[�Ƃ��ċ@�\*/

				  /*-------------------------------------------------------------------------------------*/
				  /*�����׏d�̏�����:�����_��*/
	double initial_w = 0.0002;
	//���́E���ԑw��ԁE�E�E�EW_IN_HD[HN][IN]
	W_IN_HD[0][0] = initial_w;
	W_IN_HD[0][1] = 0.078;
	W_IN_HD[1][0] = -0.423;
	W_IN_HD[1][1] = 0.009;
	//���ԁE�B��w��ԁE�E�E�EW_HD_OT[ON][HN]
	W_HD_OT[0][0] = 0.000908;
	W_HD_OT[0][1] = -0.0087;
	W_HD_OT[0][2] = -0.0048;
	W_HD_OT[0][3] = -0.0087;

	//�e臒l
	//�B��w
	CW_HD[0] = 0.0089;
	CW_HD[1] = -0.098;
	//�o�͑w
	CW_OT[0] = 0.0023;
	//�w�K�W��
	alpha = 0.7;
	beta = 0.7;
	/*-------------------------------------------------------------------------------------*/

	int loop1;		//���t���f���̎�p�̃��[�v�B
	int loop = 0;		//�w�K���[�v���̏�����
	double E_t;		//�e�p�^�[���̑��a�덷

	while (loop < times) {
		double error = 0.0;	//�w�K��loop����error��������

							/*--------------���鋳�t���f���ɑ΂��郋�[�v���n��----------------------------------*/
		for (loop1 = 0; loop1 < SAMPLINGNUM; loop1++) {
			//printf("Leaning Patern %d\n", indata[loop1].tch[0]);

			/*���`�d�̏���*/
			/*���͑w�̏o�͂��Z�b�g����*/
			for (k = 0; k < IN; k++) {
				OT_IN[k] = (double)indata[loop1].input[k];
				//printf("%f\n", OT_IN[k]);
			}

			/*���ԑw�̏o�͂����߂�*/
			for (k = 0; k < HN; k++) {		//H_1,H_2,...�����߂�
				inival = 0.0;				//�J��Ԃ��v�Z�ɂ����鏉���l�̃Z�b�g
				for (m = 0; m < IN; m++) {	//W_ji*I_i�̉��Z���ړI
					inival += (W_IN_HD[k][m])*(OT_IN[m]);
				}
				inival += CW_HD[k];			//�Ō��臒l�����Z,��inival���B��w�|�e���V����H_jv
				OT_HD[k] = sigmf(inival);	//�ʑ��֐�
			}

			/*�B��w���C���[�Ԃ̓`�d����*/
			for (i = 0; i < LAY; i++) {

			}

			/*�o�͑w�̏o�͂����߂�*/
			for (k = 0; k < ON; k++) {
				inival = 0.0;
				for (m = 0; m < HN; m++) {
					inival += (W_HD_OT[k][m])*(OT_HD[m]);
				}
				inival += CW_OT[k];			//�Ō��臒l�����Z,��inival���o�͑w�|�e���V����S_k
				OT_OT[k] = sigmf(inival);	//�ʑ��V�O���C�h�֐�
			}

			/*�t�`�d�̏���*/
			/*�덷(�o�͑w)�̎Z�o*/
			double wk, wkb;									//wk�͏o�͐M��O_k���Awkb��O_k��T_k�Ƃ̍����Z�o����̂ɗp����

			for (m = 0; m < ON; m++) {						//DEL_OT,���Ȃ킿HD-OT�Ԃ̌덷���v�Z
				TEACH[m] = (double)indata[loop1].tch[m];	//loop�P�ɂ����鋳�t�o�͂������Ă���
				wk = OT_OT[m];
				wkb = TEACH[m] - OT_OT[m];	//���t�M���Əo�͂Ƃ̍�(��_k = T_k - O_k)
				error += (wkb)*(wkb) / 2;	//�o�͑wm�ɂ����镽�ϓ��덷,��E_mp,m=0������Z
				DEL_OT[m] = wkb * ru0 * wk * (1.0 - wk);
			}
			/*�덷(�B��w)�̎Z�o*/
			for (k = 0; k < HN; k++) {	//DEL_HD,���Ȃ킿HD-IN�Ԃ̌덷���v�Z
				inival = 0.0;
				for (m = 0; m < ON; m++) {
					inival += (DEL_OT[m] * W_HD_OT[m][k]);
				}
				wk = OT_HD[k];			//����Ƃ͈Ⴂ�A���x��HD�̏o�͐M��O_j
				DEL_HD[k] = inival * ru0 * wk * (1.0 - wk);
			}
			/*�����׏d�̍X�V*/
			/*�B��w-�o�͑w��*/
			for (k = 0; k < ON; k++) {
				for (m = 0; m < HN; m++) {
					W_HD_OT[k][m] += (alpha * DEL_OT[k] * OT_HD[m]);	//�����׏dV_kj�̍X�V
				}
				CW_OT[k] += (beta * DEL_OT[k]);				//offset�̍X�V
			}
			/*���͑w-�B��w��*/
			for (k = 0; k < HN; k++) {
				for (m = 0; m < IN; m++) {
					W_IN_HD[k][m] += (alpha * DEL_HD[k] * OT_IN[m]);
				}
				CW_HD[k] += (beta * DEL_HD[k]);
			}
		}
		E_t = error;			//E_t�ɍ��܂ł̂��ׂẴp�^�[���̌덷�������n��

								/*--------------------���鋳�t���f���ɑ΂��郋�[�v���I���A���̋��t���f���̎��------------------------------*/

								/*-------------------�p�����[�^���ڂ��ϑ�����o�͂��s��------------------------------*/
		SamplingValueChangeNew(loop, E_t);
		/*-----------------------------------------------------------------------------------*/

		loop++;//�w�K�񐔂̍X�V
	}
	/*-------------------------------------�w�K�̏I��-----------------------------------------------------------*/

	/*-------------------������̃p�����[�^��f���o��-----------------------------------*/
	forGraphplot();
	/*----------------------------------------------------------------------------------*/
	return 0;
}
