using System;
using System.Runtime.InteropServices;

namespace Linalg
{
    using LOGICAL = System.Int32;
    using INDEX = System.Int32;
    public struct complex_float
    {
        float re;
        float im;
    };
    public struct complex_double
    {
        double re;
        double im;
    };

    /* Callback logical functions of one, two, or three arguments are used
    *  to select eigenvalues to sort to the top left of the Schur form.
    *  The value is selected if function returns TRUE (non-zero). */
    public delegate LOGICAL S_SELECT2 (ref float a, ref float b);
    public delegate LOGICAL S_SELECT3 (ref float a, ref float b, ref float c);
    public delegate LOGICAL D_SELECT2 (ref double a, ref double b);
    public delegate LOGICAL D_SELECT3 (ref double a, ref double b, ref double c);
    public delegate LOGICAL C_SELECT1 (ref complex_float a);
    public delegate LOGICAL C_SELECT2 (ref complex_float a, ref complex_float b);
    public delegate LOGICAL Z_SELECT1 (ref complex_double a);
    public delegate LOGICAL Z_SELECT2 (ref complex_double a, ref complex_double b);

    public enum LAYOUT { RowMajor = 101, ColMajor = 102 };
    public enum TRANSPOSE { NoTrans = 111, Trans = 112, ConjTrans = 113 };
    public enum UPLO { Upper = 121, Lower = 122 };
    public enum DIAG { NonUnit = 131, Unit = 132 };
    public enum SIDE { Left = 141, Right = 142 };

    public static class BLAS
    {
        /*                                                         
        * ===========================================================================                                                       
        * Prototypes for level 1 BLAS functions (complex are recast as routines)                                             
        * ===========================================================================                                                       
        */

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dcabs1")]
        public static extern double DCABS1(ref complex_double z);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_scabs1")]
        public static extern float SCABS1(ref complex_float c);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sdsdot")]
        public static extern float SDSDOT(ref int N, ref float alpha, ref float X,
             ref int incX, ref float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dsdot")]
        public static extern double DSDOT(ref int N, ref float X, ref int incX, ref float Y,
               ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sdot")]
        public static extern float SDOT(ref int N, ref float X, ref int incX,
              ref float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ddot")]
        public static extern double DDOT(ref int N, ref double X, ref int incX,
              ref double Y, ref int incY);

        /*                                                         
         * Functions having PrefIXES Z and C only                                                 
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cdotu_sub")]
        public static extern void CDOTU_SUB(ref int N, ref complex_float X, ref int incX,
                ref complex_float Y, ref int incY, ref complex_float dotu);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cdotc_sub")]
        public static extern void CDOTC_SUB(ref int N, ref complex_float X, ref int incX,
                ref complex_float Y, ref int incY, ref complex_float dotc);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zdotu_sub")]
        public static extern void ZDOTU_SUB(ref int N, ref complex_double X, ref int incX,
                ref complex_double Y, ref int incY, ref complex_double dotu);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zdotc_sub")]
        public static extern void ZDOTC_SUB(ref int N, ref complex_double X, ref int incX,
                ref complex_double Y, ref int incY, ref complex_double dotc);


        /*                                                         
         * Functions having PrefIXES S D SC DZ                                                 
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_snrm2")]
        public static extern float SNRM2(ref int N, ref float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sasum")]
        public static extern float SASUM(ref int N, ref float X, ref int incX);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dnrm2")]
        public static extern double DNRM2(ref int N, ref double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dasum")]
        public static extern double DASUM(ref int N, ref double X, ref int incX);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_scnrm2")]
        public static extern float SCNRM2(ref int N, ref complex_float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_scasum")]
        public static extern float SCASUM(ref int N, ref complex_float X, ref int incX);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dznrm2")]
        public static extern double DZNRM2(ref int N, ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dzasum")]
        public static extern double DZASUM(ref int N, ref complex_double X, ref int incX);


        /*                                                         
         * Functions having STANDARD 4 prefixes (S D C Z)                                               
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_isamax")]
        public static extern INDEX ISAMAX(ref int N, ref float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_idamax")]
        public static extern INDEX IDAMAX(ref int N, ref double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_icamax")]
        public static extern INDEX ICAMAX(ref int N, ref complex_float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_izamax")]
        public static extern INDEX IZAMAX(ref int N, ref complex_double X, ref int incX);

        /*                                                         
         * ===========================================================================                                                       
         * Prototypes for LEVEL 1 BLAS routines                                                  
         * ===========================================================================                                                       
         */

        /*                                                         
         * Routines with STANDARD 4 prefixes (s, d, c, z)                                               
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sswap")]
        public static extern void SSWAP(ref int N, ref float X, ref int incX,
             ref float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_scopy")]
        public static extern void SCOPY(ref int N, ref float X, ref int incX,
             ref float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_saxpy")]
        public static extern void SAXPY(ref int N, ref float alpha, ref float X,
             ref int incX, ref float Y, ref int incY);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dswap")]
        public static extern void DSWAP(ref int N, ref double X, ref int incX,
             ref double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dcopy")]
        public static extern void DCOPY(ref int N, ref double X, ref int incX,
             ref double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_daxpy")]
        public static extern void DAXPY(ref int N, ref double alpha, ref double X,
             ref int incX, ref double Y, ref int incY);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cswap")]
        public static extern void CSWAP(ref int N, ref complex_float X, ref int incX,
             ref complex_float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ccopy")]
        public static extern void CCOPY(ref int N, ref complex_float X, ref int incX,
             ref complex_float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_caxpy")]
        public static extern void CAXPY(ref int N, ref complex_float alpha, ref complex_float X,
             ref int incX, ref complex_float Y, ref int incY);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zswap")]
        public static extern void ZSWAP(ref int N, ref complex_double X, ref int incX,
             ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zcopy")]
        public static extern void ZCOPY(ref int N, ref complex_double X, ref int incX,
             ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zaxpy")]
        public static extern void ZAXPY(ref int N, ref complex_double alpha, ref complex_double X,
             ref int incX, ref complex_double Y, ref int incY);


        /*                                                         
         * Routines with s and D prefix only                                                 
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_srotg")]
        public static extern void SROTG(ref float a, ref float b, ref float c, ref float s);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_srotmg")]
        public static extern void SROTMG(ref float d1, ref float d2, ref float b1, ref float b2, ref float P);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_srot")]
        public static extern void SROT(ref int N, ref float X, ref int incX,
            ref float Y, ref int incY, ref float c, ref float s);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_srotm")]
        public static extern void SROTM(ref int N, ref float X, ref int incX,
            ref float Y, ref int incY, ref float P);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_drotg")]
        public static extern void DROTG(ref double a, ref double b, ref double c, ref double s);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_drotmg")]
        public static extern void DROTMG(ref double d1, ref double d2, ref double b1, ref double b2, ref double P);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_drot")]
        public static extern void DROT(ref int N, ref double X, ref int incX,
            ref double Y, ref int incY, ref double c, ref double s);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_drotm")]
        public static extern void DROTM(ref int N, ref double X, ref int incX,
            ref double Y, ref int incY, ref double P);


        /*                                                         
         * Routines with s D C Z CS and ZD prefixes                                              
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sscal")]
        public static extern void SSCAL(ref int N, ref float alpha, ref float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dscal")]
        public static extern void DSCAL(ref int N, ref double alpha, ref double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cscal")]
        public static extern void CSCAL(ref int N, ref complex_float alpha, ref complex_float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zscal")]
        public static extern void ZSCAL(ref int N, ref complex_double alpha, ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_csscal")]
        public static extern void CSSCAL(ref int N, ref float alpha, ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zdscal")]
        public static extern void ZDSCAL(ref int N, ref double alpha, ref complex_double X, ref int incX);

        /*                                                         
         * ===========================================================================                                                       
         * Prototypes for LEVEL 2 BLAS                                                   
         * ===========================================================================                                                       
         */

        /*                                                         
         * Routines with STANDARD 4 prefixes (S, D, C, Z)                                               
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sgemv")]
        public static extern void SGEMV(ref LAYOUT layout,
             ref TRANSPOSE TransA, ref int M, ref int N,
             ref float alpha, ref float A, ref int lda,
             ref float X, ref int incX, ref float beta,
             ref float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sgbmv")]
        public static extern void SGBMV(ref LAYOUT layout,
             ref TRANSPOSE TransA, ref int M, ref int N,
             ref int KL, ref int KU, ref float alpha,
             ref float A, ref int lda, ref float X,
             ref int incX, ref float beta, ref float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_strmv")]
        public static extern void STRMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref float A, ref int lda,
             ref float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_stbmv")]
        public static extern void STBMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref int K, ref float A, ref int lda,
             ref float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_stpmv")]
        public static extern void STPMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref float Ap, ref float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_strsv")]
        public static extern void STRSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref float A, ref int lda, ref float X,
             ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_stbsv")]
        public static extern void STBSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref int K, ref float A, ref int lda,
             ref float X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_stpsv")]
        public static extern void STPSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref float Ap, ref float X, ref int incX);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dgemv")]
        public static extern void DGEMV(ref LAYOUT layout,
             ref TRANSPOSE TransA, ref int M, ref int N,
             ref double alpha, ref double A, ref int lda,
             ref double X, ref int incX, ref double beta,
             ref double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dgbmv")]
        public static extern void DGBMV(ref LAYOUT layout,
             ref TRANSPOSE TransA, ref int M, ref int N,
             ref int KL, ref int KU, ref double alpha,
             ref double A, ref int lda, ref double X,
             ref int incX, ref double beta, ref double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dtrmv")]
        public static extern void DTRMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref double A, ref int lda,
             ref double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dtbmv")]
        public static extern void DTBMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref int K, ref double A, ref int lda,
             ref double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dtpmv")]
        public static extern void DTPMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref double Ap, ref double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dtrsv")]
        public static extern void DTRSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref double A, ref int lda, ref double X,
             ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dtbsv")]
        public static extern void DTBSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref int K, ref double A, ref int lda,
             ref double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dtpsv")]
        public static extern void DTPSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref double Ap, ref double X, ref int incX);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cgemv")]
        public static extern void CGEMV(ref LAYOUT layout,
             ref TRANSPOSE TransA, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double X, ref int incX, ref complex_double beta,
             ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cgbmv")]
        public static extern void CGBMV(ref LAYOUT layout,
             ref TRANSPOSE TransA, ref int M, ref int N,
             ref int KL, ref int KU, ref complex_double alpha,
             ref complex_double A, ref int lda, ref complex_double X,
             ref int incX, ref complex_double beta, ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ctrmv")]
        public static extern void CTRMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref complex_double A, ref int lda,
             ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ctbmv")]
        public static extern void CTBMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref int K, ref complex_double A, ref int lda,
             ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ctpmv")]
        public static extern void CTPMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref complex_double Ap, ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ctrsv")]
        public static extern void CTRSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref complex_double A, ref int lda, ref complex_double X,
             ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ctbsv")]
        public static extern void CTBSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref int K, ref complex_double A, ref int lda,
             ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ctpsv")]
        public static extern void CTPSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref complex_double Ap, ref complex_double X, ref int incX);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zgemv")]
        public static extern void ZGEMV(ref LAYOUT layout,
             ref TRANSPOSE TransA, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double X, ref int incX, ref complex_double beta,
             ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zgbmv")]
        public static extern void ZGBMV(ref LAYOUT layout,
             ref TRANSPOSE TransA, ref int M, ref int N,
             ref int KL, ref int KU, ref complex_double alpha,
             ref complex_double A, ref int lda, ref complex_double X,
             ref int incX, ref complex_double beta, ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ztrmv")]
        public static extern void ZTRMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref complex_double A, ref int lda,
             ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ztbmv")]
        public static extern void ZTBMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref int K, ref complex_double A, ref int lda,
             ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ztpmv")]
        public static extern void ZTPMV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref complex_double Ap, ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ztrsv")]
        public static extern void ZTRSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref complex_double A, ref int lda, ref complex_double X,
             ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ztbsv")]
        public static extern void ZTBSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref int K, ref complex_double A, ref int lda,
             ref complex_double X, ref int incX);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ztpsv")]
        public static extern void ZTPSV(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE TransA, ref DIAG Diag,
             ref int N, ref complex_double Ap, ref complex_double X, ref int incX);


        /*                                                         
         * Routines with s and D prefixes only                                                 
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ssymv")]
        public static extern void SSYMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref float alpha, ref float A,
             ref int lda, ref float X, ref int incX,
             ref float beta, ref float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ssbmv")]
        public static extern void SSBMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref int K, ref float alpha, ref float A,
             ref int lda, ref float X, ref int incX,
             ref float beta, ref float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sspmv")]
        public static extern void SSPMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref float alpha, ref float Ap,
             ref float X, ref int incX,
             ref float beta, ref float Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sger")]
        public static extern void SGER(ref LAYOUT layout, ref int M, ref int N,
            ref float alpha, ref float X, ref int incX,
            ref float Y, ref int incY, ref float A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ssyr")]
        public static extern void SSYR(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref float alpha, ref float X,
            ref int incX, ref float A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sspr")]
        public static extern void SSPR(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref float alpha, ref float X,
            ref int incX, ref float Ap);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ssyr2")]
        public static extern void SSYR2(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref float alpha, ref float X,
            ref int incX, ref float Y, ref int incY, ref float A,
            ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sspr2")]
        public static extern void SSPR2(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref float alpha, ref float X,
            ref int incX, ref float Y, ref int incY, ref float A);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dsymv")]
        public static extern void DSYMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref double alpha, ref double A,
             ref int lda, ref double X, ref int incX,
             ref double beta, ref double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dsbmv")]
        public static extern void DSBMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref int K, ref double alpha, ref double A,
             ref int lda, ref double X, ref int incX,
             ref double beta, ref double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dspmv")]
        public static extern void DSPMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref double alpha, ref double Ap,
             ref double X, ref int incX,
             ref double beta, ref double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dger")]
        public static extern void DGER(ref LAYOUT layout, ref int M, ref int N,
            ref double alpha, ref double X, ref int incX,
            ref double Y, ref int incY, ref double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dsyr")]
        public static extern void DSYR(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref double alpha, ref double X,
            ref int incX, ref double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dspr")]
        public static extern void DSPR(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref double alpha, ref double X,
            ref int incX, ref double Ap);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dsyr2")]
        public static extern void DSYR2(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref double alpha, ref double X,
            ref int incX, ref double Y, ref int incY, ref double A,
            ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dspr2")]
        public static extern void DSPR2(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref double alpha, ref double X,
            ref int incX, ref double Y, ref int incY, ref double A);


        /*                                                         
         * Routines with c and Z prefixes only                                                 
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_chemv")]
        public static extern void CHEMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref complex_double alpha, ref complex_double A,
             ref int lda, ref complex_double X, ref int incX,
             ref complex_double beta, ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_chbmv")]
        public static extern void CHBMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref int K, ref complex_double alpha, ref complex_double A,
             ref int lda, ref complex_double X, ref int incX,
             ref complex_double beta, ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_chpmv")]
        public static extern void CHPMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref complex_double alpha, ref complex_double Ap,
             ref complex_double X, ref int incX,
             ref complex_double beta, ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cgeru")]
        public static extern void CGERU(ref LAYOUT layout, ref int M, ref int N,
             ref complex_double alpha, ref complex_double X, ref int incX,
             ref complex_double Y, ref int incY, ref complex_double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cgerc")]
        public static extern void CGERC(ref LAYOUT layout, ref int M, ref int N,
             ref complex_double alpha, ref complex_double X, ref int incX,
             ref complex_double Y, ref int incY, ref complex_double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cher")]
        public static extern void CHER(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref float alpha, ref complex_double X, ref int incX,
            ref complex_double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_chpr")]
        public static extern void CHPR(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref float alpha, ref complex_double X,
            ref int incX, ref complex_double A);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cher2")]
        public static extern void CHER2(ref LAYOUT layout, ref UPLO Uplo, ref int N,
            ref complex_double alpha, ref complex_double X, ref int incX,
            ref complex_double Y, ref int incY, ref complex_double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_chpr2")]
        public static extern void CHPR2(ref LAYOUT layout, ref UPLO Uplo, ref int N,
            ref complex_double alpha, ref complex_double X, ref int incX,
            ref complex_double Y, ref int incY, ref complex_double Ap);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zhemv")]
        public static extern void ZHEMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref complex_double alpha, ref complex_double A,
             ref int lda, ref complex_double X, ref int incX,
             ref complex_double beta, ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zhbmv")]
        public static extern void ZHBMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref int K, ref complex_double alpha, ref complex_double A,
             ref int lda, ref complex_double X, ref int incX,
             ref complex_double beta, ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zhpmv")]
        public static extern void ZHPMV(ref LAYOUT layout, ref UPLO Uplo,
             ref int N, ref complex_double alpha, ref complex_double Ap,
             ref complex_double X, ref int incX,
             ref complex_double beta, ref complex_double Y, ref int incY);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zgeru")]
        public static extern void ZGERU(ref LAYOUT layout, ref int M, ref int N,
             ref complex_double alpha, ref complex_double X, ref int incX,
             ref complex_double Y, ref int incY, ref complex_double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zgerc")]
        public static extern void ZGERC(ref LAYOUT layout, ref int M, ref int N,
             ref complex_double alpha, ref complex_double X, ref int incX,
             ref complex_double Y, ref int incY, ref complex_double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zher")]
        public static extern void ZHER(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref double alpha, ref complex_double X, ref int incX,
            ref complex_double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zhpr")]
        public static extern void ZHPR(ref LAYOUT layout, ref UPLO Uplo,
            ref int N, ref double alpha, ref complex_double X,
            ref int incX, ref complex_double A);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zher2")]
        public static extern void ZHER2(ref LAYOUT layout, ref UPLO Uplo, ref int N,
            ref complex_double alpha, ref complex_double X, ref int incX,
            ref complex_double Y, ref int incY, ref complex_double A, ref int lda);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zhpr2")]
        public static extern void ZHPR2(ref LAYOUT layout, ref UPLO Uplo, ref int N,
            ref complex_double alpha, ref complex_double X, ref int incX,
            ref complex_double Y, ref int incY, ref complex_double Ap);

        /*                                                         
         * ===========================================================================                                                       
         * Prototypes for LEVEL 3 BLAS                                                   
         * ===========================================================================                                                       
         */

        /*                                                         
         * Routines with STANDARD 4 prefixes (S, D, C, Z)                                               
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_sgemm")]
        public static extern void SGEMM(ref LAYOUT layout, ref TRANSPOSE TransA,
             ref TRANSPOSE TransB, ref int M, ref int N,
             ref int K, ref float alpha, ref float A,
             ref int lda, ref float B, ref int ldb,
             ref float beta, ref float C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ssymm")]
        public static extern void SSYMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref int M, ref int N,
             ref float alpha, ref float A, ref int lda,
             ref float B, ref int ldb, ref float beta,
             ref float C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ssyrk")]
        public static extern void SSYRK(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE Trans, ref int N, ref int K,
             ref float alpha, ref float A, ref int lda,
             ref float beta, ref float C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ssyr2k")]
        public static extern void SSYR2K(ref LAYOUT layout, ref UPLO Uplo,
              ref TRANSPOSE Trans, ref int N, ref int K,
              ref float alpha, ref float A, ref int lda,
              ref float B, ref int ldb, ref float beta,
              ref float C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_strmm")]
        public static extern void STRMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref TRANSPOSE TransA,
             ref DIAG Diag, ref int M, ref int N,
             ref float alpha, ref float A, ref int lda,
             ref float B, ref int ldb);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_strsm")]
        public static extern void STRSM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref TRANSPOSE TransA,
             ref DIAG Diag, ref int M, ref int N,
             ref float alpha, ref float A, ref int lda,
             ref float B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dgemm")]
        public static extern void DGEMM(ref LAYOUT layout, ref TRANSPOSE TransA,
             ref TRANSPOSE TransB, ref int M, ref int N,
             ref int K, ref double alpha, ref double A,
             ref int lda, ref double B, ref int ldb,
             ref double beta, ref double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dsymm")]
        public static extern void DSYMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref int M, ref int N,
             ref double alpha, ref double A, ref int lda,
             ref double B, ref int ldb, ref double beta,
             ref double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dsyrk")]
        public static extern void DSYRK(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE Trans, ref int N, ref int K,
             ref double alpha, ref double A, ref int lda,
             ref double beta, ref double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dsyr2k")]
        public static extern void DSYR2K(ref LAYOUT layout, ref UPLO Uplo,
              ref TRANSPOSE Trans, ref int N, ref int K,
              ref double alpha, ref double A, ref int lda,
              ref double B, ref int ldb, ref double beta,
              ref double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dtrmm")]
        public static extern void DTRMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref TRANSPOSE TransA,
             ref DIAG Diag, ref int M, ref int N,
             ref double alpha, ref double A, ref int lda,
             ref double B, ref int ldb);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_dtrsm")]
        public static extern void DTRSM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref TRANSPOSE TransA,
             ref DIAG Diag, ref int M, ref int N,
             ref double alpha, ref double A, ref int lda,
             ref double B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cgemm")]
        public static extern void CGEMM(ref LAYOUT layout, ref TRANSPOSE TransA,
             ref TRANSPOSE TransB, ref int M, ref int N,
             ref int K, ref complex_double alpha, ref complex_double A,
             ref int lda, ref complex_double B, ref int ldb,
             ref complex_double beta, ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_csymm")]
        public static extern void CSYMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double B, ref int ldb, ref complex_double beta,
             ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_csyrk")]
        public static extern void CSYRK(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE Trans, ref int N, ref int K,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double beta, ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_csyr2k")]
        public static extern void CSYR2K(ref LAYOUT layout, ref UPLO Uplo,
              ref TRANSPOSE Trans, ref int N, ref int K,
              ref complex_double alpha, ref complex_double A, ref int lda,
              ref complex_double B, ref int ldb, ref complex_double beta,
              ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ctrmm")]
        public static extern void CTRMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref TRANSPOSE TransA,
             ref DIAG Diag, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double B, ref int ldb);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ctrsm")]
        public static extern void CTRSM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref TRANSPOSE TransA,
             ref DIAG Diag, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zgemm")]
        public static extern void ZGEMM(ref LAYOUT layout, ref TRANSPOSE TransA,
             ref TRANSPOSE TransB, ref int M, ref int N,
             ref int K, ref complex_double alpha, ref complex_double A,
             ref int lda, ref complex_double B, ref int ldb,
             ref complex_double beta, ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zsymm")]
        public static extern void ZSYMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double B, ref int ldb, ref complex_double beta,
             ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zsyrk")]
        public static extern void ZSYRK(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE Trans, ref int N, ref int K,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double beta, ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zsyr2k")]
        public static extern void ZSYR2K(ref LAYOUT layout, ref UPLO Uplo,
              ref TRANSPOSE Trans, ref int N, ref int K,
              ref complex_double alpha, ref complex_double A, ref int lda,
              ref complex_double B, ref int ldb, ref complex_double beta,
              ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ztrmm")]
        public static extern void ZTRMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref TRANSPOSE TransA,
             ref DIAG Diag, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double B, ref int ldb);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_ztrsm")]
        public static extern void ZTRSM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref TRANSPOSE TransA,
             ref DIAG Diag, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double B, ref int ldb);


        /*                                                         
         * Routines with PrefIXES C and Z only                                                 
         */
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_chemm")]
        public static extern void CHEMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double B, ref int ldb, ref complex_double beta,
             ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cherk")]
        public static extern void CHERK(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE Trans, ref int N, ref int K,
             ref float alpha, ref complex_double A, ref int lda,
             ref float beta, ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_cher2k")]
        public static extern void CHER2K(ref LAYOUT layout, ref UPLO Uplo,
              ref TRANSPOSE Trans, ref int N, ref int K,
              ref complex_double alpha, ref complex_double A, ref int lda,
              ref complex_double B, ref int ldb, ref float beta,
              ref complex_double C, ref int ldc);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zhemm")]
        public static extern void ZHEMM(ref LAYOUT layout, ref SIDE Side,
             ref UPLO Uplo, ref int M, ref int N,
             ref complex_double alpha, ref complex_double A, ref int lda,
             ref complex_double B, ref int ldb, ref complex_double beta,
             ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zherk")]
        public static extern void ZHERK(ref LAYOUT layout, ref UPLO Uplo,
             ref TRANSPOSE Trans, ref int N, ref int K,
             ref double alpha, ref complex_double A, ref int lda,
             ref double beta, ref complex_double C, ref int ldc);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cblas_zher2k")]
        public static extern void ZHER2K(ref LAYOUT layout, ref UPLO Uplo,
              ref TRANSPOSE Trans, ref int N, ref int K,
              ref complex_double alpha, ref complex_double A, ref int lda,
              ref complex_double B, ref int ldb, ref double beta,
              ref complex_double C, ref int ldc);
    }

    public static class LAPACK
    {

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cbbcsd_")]
        public static extern void CBBCSD(
         ref char jobu1, ref char jobu2, ref char jobv1t, ref char jobv2t, ref char trans,
         ref int m, ref int p, ref int q,
         ref float theta,
         ref float phi,
         ref complex_float U1, ref int ldu1,
         ref complex_float U2, ref int ldu2,
         ref complex_float V1T, ref int ldv1t,
         ref complex_float V2T, ref int ldv2t,
         ref float B11D,
         ref float B11E,
         ref float B12D,
         ref float B12E,
         ref float B21D,
         ref float B21E,
         ref float B22D,
         ref float B22E,
         ref float rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dbbcsd_")]
        public static extern void DBBCSD(
         ref char jobu1, ref char jobu2, ref char jobv1t, ref char jobv2t, ref char trans,
         ref int m, ref int p, ref int q,
         ref double theta,
         ref double phi,
         ref double U1, ref int ldu1,
         ref double U2, ref int ldu2,
         ref double V1T, ref int ldv1t,
         ref double V2T, ref int ldv2t,
         ref double B11D,
         ref double B11E,
         ref double B12D,
         ref double B12E,
         ref double b21d,
         ref double b21e,
         ref double b22d,
         ref double b22e,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sbbcsd_")]
        public static extern void SBBCSD(
         ref char jobu1, ref char jobu2, ref char jobv1t, ref char jobv2t, ref char trans,
         ref int m, ref int p, ref int q,
         ref float theta,
         ref float phi,
         ref float U1, ref int ldu1,
         ref float U2, ref int ldu2,
         ref float V1T, ref int ldv1t,
         ref float V2T, ref int ldv2t,
         ref float B11D,
         ref float B11E,
         ref float B12D,
         ref float B12E,
         ref float B21D,
         ref float B21E,
         ref float B22D,
         ref float B22E,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zbbcsd_")]
        public static extern void ZBBCSD(
         ref char jobu1, ref char jobu2, ref char jobv1t, ref char jobv2t, ref char trans,
         ref int m, ref int p, ref int q,
         ref double theta,
         ref double phi,
         ref complex_double U1, ref int ldu1,
         ref complex_double U2, ref int ldu2,
         ref complex_double V1T, ref int ldv1t,
         ref complex_double V2T, ref int ldv2t,
         ref double B11D,
         ref double B11E,
         ref double B12D,
         ref double B12E,
         ref double B21D,
         ref double B21E,
         ref double B22D,
         ref double B22E,
         ref double rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dbdsdc_")]
        public static extern void DBDSDC(
         ref char uplo, ref char compq,
         ref int n,
         ref double D,
         ref double E,
         ref double U, ref int ldu,
         ref double VT, ref int ldvt,
         ref double Q, ref int IQ,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sbdsdc_")]
        public static extern void SBDSDC(
         ref char uplo, ref char compq,
         ref int n,
         ref float D,
         ref float E,
         ref float U, ref int ldu,
         ref float VT, ref int ldvt,
         ref float Q, ref int IQ,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cbdsqr_")]
        public static extern void CBDSQR(
         ref char uplo,
         ref int n, ref int ncvt, ref int nru, ref int ncc,
         ref float D,
         ref float E,
         ref complex_float VT, ref int ldvt,
         ref complex_float U, ref int ldu,
         ref complex_float C, ref int ldc,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dbdsqr_")]
        public static extern void DBDSQR(
         ref char uplo,
         ref int n, ref int ncvt, ref int nru, ref int ncc,
         ref double D,
         ref double E,
         ref double VT, ref int ldvt,
         ref double U, ref int ldu,
         ref double C, ref int ldc,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sbdsqr_")]
        public static extern void SBDSQR(
         ref char uplo,
         ref int n, ref int ncvt, ref int nru, ref int ncc,
         ref float D,
         ref float E,
         ref float VT, ref int ldvt,
         ref float U, ref int ldu,
         ref float C, ref int ldc,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zbdsqr_")]
        public static extern void ZBDSQR(
         ref char uplo,
         ref int n, ref int ncvt, ref int nru, ref int ncc,
         ref double D,
         ref double E,
         ref complex_double VT, ref int ldvt,
         ref complex_double U, ref int ldu,
         ref complex_double C, ref int ldc,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dbdsvdx_")]
        public static extern void DBDSVDX(
         ref char uplo, ref char jobz, ref char range,
         ref int n,
         ref double D,
         ref double E,
         ref double vl,
         ref double vu, ref int il, ref int iu, ref int ns,
         ref double S,
         ref double Z, ref int ldz,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sbdsvdx_")]
        public static extern void SBDSVDX(
         ref char uplo, ref char jobz, ref char range,
         ref int n,
         ref float D,
         ref float E,
         ref float vl,
         ref float vu, ref int il, ref int iu, ref int ns,
         ref float S,
         ref float Z, ref int ldz,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ddisna_")]
        public static extern void DDISNA(
         ref char job,
         ref int m, ref int n,
         ref double D,
         ref double SEP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sdisna_")]
        public static extern void SDISNA(
         ref char job,
         ref int m, ref int n,
         ref float D,
         ref float SEP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbbrd_")]
        public static extern void CGBBRD(
         ref char vect,
         ref int m, ref int n, ref int ncc, ref int kl, ref int ku,
         ref complex_float AB, ref int ldab,
         ref float D,
         ref float E,
         ref complex_float Q, ref int ldq,
         ref complex_float PT, ref int ldpt,
         ref complex_float C, ref int ldc,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbbrd_")]
        public static extern void DGBBRD(
         ref char vect,
         ref int m, ref int n, ref int ncc, ref int kl, ref int ku,
         ref double AB, ref int ldab,
         ref double D,
         ref double E,
         ref double Q, ref int ldq,
         ref double PT, ref int ldpt,
         ref double C, ref int ldc,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbbrd_")]
        public static extern void SGBBRD(
         ref char vect,
         ref int m, ref int n, ref int ncc, ref int kl, ref int ku,
         ref float AB, ref int ldab,
         ref float D,
         ref float E,
         ref float Q, ref int ldq,
         ref float PT, ref int ldpt,
         ref float C, ref int ldc,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbbrd_")]
        public static extern void ZGBBRD(
         ref char vect,
         ref int m, ref int n, ref int ncc, ref int kl, ref int ku,
         ref complex_double AB, ref int ldab,
         ref double D,
         ref double E,
         ref complex_double Q, ref int ldq,
         ref complex_double PT, ref int ldpt,
         ref complex_double C, ref int ldc,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbcon_")]
        public static extern void CGBCON(
         ref char norm,
         ref int n, ref int kl, ref int ku,
         ref complex_float AB, ref int ldab, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbcon_")]
        public static extern void DGBCON(
         ref char norm,
         ref int n, ref int kl, ref int ku,
         ref double AB, ref int ldab, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbcon_")]
        public static extern void SGBCON(
         ref char norm,
         ref int n, ref int kl, ref int ku,
         ref float AB, ref int ldab, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbcon_")]
        public static extern void ZGBCON(
         ref char norm,
         ref int n, ref int kl, ref int ku,
         ref complex_double AB, ref int ldab, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbequ_")]
        public static extern void CGBEQU(
         ref int m, ref int n, ref int kl, ref int ku,
         ref complex_float AB, ref int ldab,
         ref float R,
         ref float C,
         ref float rowcnd,
         ref float colcnd,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbequ_")]
        public static extern void DGBEQU(
         ref int m, ref int n, ref int kl, ref int ku,
         ref double AB, ref int ldab,
         ref double R,
         ref double C,
         ref double rowcnd,
         ref double colcnd,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbequ_")]
        public static extern void SGBEQU(
         ref int m, ref int n, ref int kl, ref int ku,
         ref float AB, ref int ldab,
         ref float R,
         ref float C,
         ref float rowcnd,
         ref float colcnd,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbequ_")]
        public static extern void ZGBEQU(
         ref int m, ref int n, ref int kl, ref int ku,
         ref complex_double AB, ref int ldab,
         ref double R,
         ref double C,
         ref double rowcnd,
         ref double colcnd,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbequb_")]
        public static extern void CGBEQUB(
         ref int m, ref int n, ref int kl, ref int ku,
         ref complex_float AB, ref int ldab,
         ref float R,
         ref float C,
         ref float rowcnd,
         ref float colcnd,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbequb_")]
        public static extern void DGBEQUB(
         ref int m, ref int n, ref int kl, ref int ku,
         ref double AB, ref int ldab,
         ref double R,
         ref double C,
         ref double rowcnd,
         ref double colcnd,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbequb_")]
        public static extern void SGBEQUB(
         ref int m, ref int n, ref int kl, ref int ku,
         ref float AB, ref int ldab,
         ref float R,
         ref float C,
         ref float rowcnd,
         ref float colcnd,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbequb_")]
        public static extern void ZGBEQUB(
         ref int m, ref int n, ref int kl, ref int ku,
         ref complex_double AB, ref int ldab,
         ref double R,
         ref double C,
         ref double rowcnd,
         ref double colcnd,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbrfs_")]
        public static extern void CGBRFS(
         ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float AFB, ref int ldafb, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbrfs_")]
        public static extern void DGBRFS(
         ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref double AB, ref int ldab,
         ref double AFB, ref int ldafb, ref int ipiv,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbrfs_")]
        public static extern void SGBRFS(
         ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref float AB, ref int ldab,
         ref float AFB, ref int ldafb, ref int ipiv,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbrfs_")]
        public static extern void ZGBRFS(
         ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double AFB, ref int ldafb, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbrfsx_")]
        public static extern void CGBRFSX(
         ref char trans, ref char equed,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float AFB, ref int ldafb, ref int ipiv,
         ref float R,
         ref float C,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbrfsx_")]
        public static extern void DGBRFSX(
         ref char trans, ref char equed,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref double AB, ref int ldab,
         ref double AFB, ref int ldafb, ref int ipiv,
         ref double R,
         ref double C,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbrfsx_")]
        public static extern void SGBRFSX(
         ref char trans, ref char equed,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref float AB, ref int ldab,
         ref float AFB, ref int ldafb, ref int ipiv,
         ref float R,
         ref float C,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbrfsx_")]
        public static extern void ZGBRFSX(
         ref char trans, ref char equed,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double AFB, ref int ldafb, ref int ipiv,
         ref double R,
         ref double C,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbsv_")]
        public static extern void CGBSV(
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_float AB, ref int ldab, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbsv_")]
        public static extern void DGBSV(
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref double AB, ref int ldab, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbsv_")]
        public static extern void SGBSV(
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref float AB, ref int ldab, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbsv_")]
        public static extern void ZGBSV(
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_double AB, ref int ldab, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbsvx_")]
        public static extern void CGBSVX(
         ref char fact, ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float AFB, ref int ldafb, ref int ipiv, ref char equed,
         ref float R,
         ref float C,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbsvx_")]
        public static extern void DGBSVX(
         ref char fact, ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref double AB, ref int ldab,
         ref double AFB, ref int ldafb, ref int ipiv, ref char equed,
         ref double R,
         ref double C,
         ref double B,
         ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbsvx_")]
        public static extern void SGBSVX(
         ref char fact, ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref float AB, ref int ldab,
         ref float AFB, ref int ldafb, ref int ipiv, ref char equed,
         ref float R,
         ref float C,
         ref float B,
         ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbsvx_")]
        public static extern void ZGBSVX(
         ref char fact, ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double AFB, ref int ldafb, ref int ipiv, ref char equed,
         ref double R,
         ref double C,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbsvxx_")]
        public static extern void CGBSVXX(
         ref char fact, ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float AFB, ref int ldafb, ref int ipiv, ref char equed,
         ref float R,
         ref float C,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float rpvgrw,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbsvxx_")]
        public static extern void DGBSVXX(
         ref char fact, ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref double AB, ref int ldab,
         ref double AFB, ref int ldafb, ref int ipiv, ref char equed,
         ref double R,
         ref double C,
         ref double B,
         ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double rpvgrw,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbsvxx_")]
        public static extern void SGBSVXX(
         ref char fact, ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref float AB, ref int ldab,
         ref float AFB, ref int ldafb, ref int ipiv, ref char equed,
         ref float R,
         ref float C,
         ref float B,
         ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float rpvgrw,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbsvxx_")]
        public static extern void ZGBSVXX(
         ref char fact, ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double AFB, ref int ldafb, ref int ipiv, ref char equed,
         ref double R,
         ref double C,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double rpvgrw,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbtrf_")]
        public static extern void CGBTRF(
         ref int m, ref int n, ref int kl, ref int ku,
         ref complex_float AB, ref int ldab, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbtrf_")]
        public static extern void DGBTRF(
         ref int m, ref int n, ref int kl, ref int ku,
         ref double AB, ref int ldab, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbtrf_")]
        public static extern void SGBTRF(
         ref int m, ref int n, ref int kl, ref int ku,
         ref float AB, ref int ldab, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbtrf_")]
        public static extern void ZGBTRF(
         ref int m, ref int n, ref int kl, ref int ku,
         ref complex_double AB, ref int ldab, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgbtrs_")]
        public static extern void CGBTRS(
         ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_float AB, ref int ldab, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgbtrs_")]
        public static extern void DGBTRS(
         ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref double AB, ref int ldab, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgbtrs_")]
        public static extern void SGBTRS(
         ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref float AB, ref int ldab, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgbtrs_")]
        public static extern void ZGBTRS(
         ref char trans,
         ref int n, ref int kl, ref int ku, ref int nrhs,
         ref complex_double AB, ref int ldab, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgebak_")]
        public static extern void CGEBAK(
         ref char job, ref char side,
         ref int n, ref int ilo, ref int ihi,
         ref float scale, ref int m,
         ref complex_float V, ref int ldv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgebak_")]
        public static extern void DGEBAK(
         ref char job, ref char side,
         ref int n, ref int ilo, ref int ihi,
         ref double scale, ref int m,
         ref double V, ref int ldv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgebak_")]
        public static extern void SGEBAK(
         ref char job, ref char side,
         ref int n, ref int ilo, ref int ihi,
         ref float scale, ref int m,
         ref float V, ref int ldv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgebak_")]
        public static extern void ZGEBAK(
         ref char job, ref char side,
         ref int n, ref int ilo, ref int ihi,
         ref double scale, ref int m,
         ref complex_double V, ref int ldv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgebal_")]
        public static extern void CGEBAL(
         ref char job,
         ref int n,
         ref complex_float A, ref int lda, ref int ilo, ref int ihi,
         ref float scale,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgebal_")]
        public static extern void DGEBAL(
         ref char job,
         ref int n,
         ref double A, ref int lda, ref int ilo, ref int ihi,
         ref double scale,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgebal_")]
        public static extern void SGEBAL(
         ref char job,
         ref int n,
         ref float A, ref int lda, ref int ilo, ref int ihi,
         ref float scale,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgebal_")]
        public static extern void ZGEBAL(
         ref char job,
         ref int n,
         ref complex_double A, ref int lda, ref int ilo, ref int ihi,
         ref double scale,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgebrd_")]
        public static extern void CGEBRD(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float D,
         ref float E,
         ref complex_float tauq,
         ref complex_float taup,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgebrd_")]
        public static extern void DGEBRD(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double D,
         ref double E,
         ref double tauq,
         ref double taup,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgebrd_")]
        public static extern void SGEBRD(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float D,
         ref float E,
         ref float tauq,
         ref float taup,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgebrd_")]
        public static extern void ZGEBRD(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double D,
         ref double E,
         ref complex_double tauq,
         ref complex_double taup,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgecon_")]
        public static extern void CGECON(
         ref char norm,
         ref int n,
         ref complex_float A, ref int lda,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgecon_")]
        public static extern void DGECON(
         ref char norm,
         ref int n,
         ref double A, ref int lda,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgecon_")]
        public static extern void SGECON(
         ref char norm,
         ref int n,
         ref float A, ref int lda,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgecon_")]
        public static extern void ZGECON(
         ref char norm,
         ref int n,
         ref complex_double A, ref int lda,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref double rwork,

         ref int info);
        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeequ_")]
        public static extern void CGEEQU(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float R,
         ref float C,
         ref float rowcnd,
         ref float colcnd,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeequ_")]
        public static extern void DGEEQU(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double R,
         ref double C,
         ref double rowcnd,
         ref double colcnd,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeequ_")]
        public static extern void SGEEQU(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float R,
         ref float C,
         ref float rowcnd,
         ref float colcnd,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeequ_")]
        public static extern void ZGEEQU(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double R,
         ref double C,
         ref double rowcnd,
         ref double colcnd,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeequb_")]
        public static extern void CGEEQUB(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float R,
         ref float C,
         ref float rowcnd,
         ref float colcnd,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeequb_")]
        public static extern void DGEEQUB(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double R,
         ref double C,
         ref double rowcnd,
         ref double colcnd,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeequb_")]
        public static extern void SGEEQUB(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float R,
         ref float C,
         ref float rowcnd,
         ref float colcnd,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeequb_")]
        public static extern void ZGEEQUB(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double R,
         ref double C,
         ref double rowcnd,
         ref double colcnd,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgees_")]
        public static extern void CGEES(
         ref char jobvs, ref char sort, ref C_SELECT1 select,
         ref int n,
         ref complex_float A, ref int lda, ref int sdim,
         ref complex_float W,
         ref complex_float VS, ref int ldvs,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgees_")]
        public static extern void DGEES(
         ref char jobvs, ref char sort, ref D_SELECT2 select,
         ref int n,
         ref double A, ref int lda, ref int sdim,
         ref double WR,
         ref double WI,
         ref double VS, ref int ldvs,
         ref double work, ref int lwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgees_")]
        public static extern void SGEES(
         ref char jobvs, ref char sort, ref S_SELECT2 select,
         ref int n,
         ref float A, ref int lda, ref int sdim,
         ref float WR,
         ref float WI,
         ref float VS, ref int ldvs,
         ref float work, ref int lwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgees_")]
        public static extern void ZGEES(
         ref char jobvs, ref char sort, ref Z_SELECT1 select,
         ref int n,
         ref complex_double A, ref int lda, ref int sdim,
         ref complex_double W,
         ref complex_double VS, ref int ldvs,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeesx_")]
        public static extern void CGEESX(
         ref char jobvs, ref char sort, ref C_SELECT1 select, ref char sense,
         ref int n,
         ref complex_float A, ref int lda, ref int sdim,
         ref complex_float W,
         ref complex_float VS, ref int ldvs,
         ref float rconde,
         ref float rcondv,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeesx_")]
        public static extern void DGEESX(
         ref char jobvs, ref char sort, ref D_SELECT2 select, ref char sense,
         ref int n,
         ref double A, ref int lda, ref int sdim,
         ref double WR,
         ref double WI,
         ref double VS, ref int ldvs,
         ref double rconde,
         ref double rcondv,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeesx_")]
        public static extern void SGEESX(
         ref char jobvs, ref char sort, ref S_SELECT2 select, ref char sense,
         ref int n,
         ref float A, ref int lda, ref int sdim,
         ref float WR,
         ref float WI,
         ref float VS, ref int ldvs,
         ref float rconde,
         ref float rcondv,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeesx_")]
        public static extern void ZGEESX(
         ref char jobvs, ref char sort, ref Z_SELECT1 select, ref char sense,
         ref int n,
         ref complex_double A, ref int lda, ref int sdim,
         ref complex_double W,
         ref complex_double VS, ref int ldvs,
         ref double rconde,
         ref double rcondv,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeev_")]
        public static extern void CGEEV(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float W,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeev_")]
        public static extern void DGEEV(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref double A, ref int lda,
         ref double WR,
         ref double WI,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeev_")]
        public static extern void SGEEV(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref float A, ref int lda,
         ref float WR,
         ref float WI,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeev_")]
        public static extern void ZGEEV(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double W,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeevx_")]
        public static extern void CGEEVX(
         ref char balanc, ref char jobvl, ref char jobvr, ref char sense,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float W,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr, ref int ilo, ref int ihi,
         ref float scale,
         ref float abnrm,
         ref float rconde,
         ref float rcondv,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeevx_")]
        public static extern void DGEEVX(
         ref char balanc, ref char jobvl, ref char jobvr, ref char sense,
         ref int n,
         ref double A, ref int lda,
         ref double WR,
         ref double WI,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr, ref int ilo, ref int ihi,
         ref double scale,
         ref double abnrm,
         ref double rconde,
         ref double rcondv,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeevx_")]
        public static extern void SGEEVX(
         ref char balanc, ref char jobvl, ref char jobvr, ref char sense,
         ref int n,
         ref float A, ref int lda,
         ref float WR,
         ref float WI,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr, ref int ilo, ref int ihi,
         ref float scale,
         ref float abnrm,
         ref float rconde,
         ref float rcondv,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeevx_")]
        public static extern void ZGEEVX(
         ref char balanc, ref char jobvl, ref char jobvr, ref char sense,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double W,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr, ref int ilo, ref int ihi,
         ref double scale,
         ref double abnrm,
         ref double rconde,
         ref double rcondv,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgehrd_")]
        public static extern void CGEHRD(
         ref int n, ref int ilo, ref int ihi,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgehrd_")]
        public static extern void DGEHRD(
         ref int n, ref int ilo, ref int ihi,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgehrd_")]
        public static extern void SGEHRD(
         ref int n, ref int ilo, ref int ihi,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgehrd_")]
        public static extern void ZGEHRD(
         ref int n, ref int ilo, ref int ihi,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgejsv_")]
        public static extern void CGEJSV(
         ref char joba, ref char jobu, ref char jobv, ref char jobr, ref char jobt, ref char jobp,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float SVA,
         ref complex_float U, ref int ldu,
         ref complex_float V, ref int ldv,
         ref complex_float cwork, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgejsv_")]
        public static extern void DGEJSV(
         ref char joba, ref char jobu, ref char jobv, ref char jobr, ref char jobt, ref char jobp,
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double SVA,
         ref double U, ref int ldu,
         ref double V, ref int ldv,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgejsv_")]
        public static extern void SGEJSV(
         ref char joba, ref char jobu, ref char jobv, ref char jobr, ref char jobt, ref char jobp,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float SVA,
         ref float U, ref int ldu,
         ref float V, ref int ldv,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgejsv_")]
        public static extern void ZGEJSV(
         ref char joba, ref char jobu, ref char jobv, ref char jobr, ref char jobt, ref char jobp,
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double SVA,
         ref complex_double U, ref int ldu,
         ref complex_double V, ref int ldv,
         ref complex_double cwork, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgelq_")]
        public static extern void CGELQ(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float T, ref int tsize,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgelq_")]
        public static extern void DGELQ(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double T, ref int tsize,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgelq_")]
        public static extern void SGELQ(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float T, ref int tsize,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgelq_")]
        public static extern void ZGELQ(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double T, ref int tsize,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgelq2_")]
        public static extern void CGELQ2(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgelq2_")]
        public static extern void DGELQ2(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgelq2_")]
        public static extern void SGELQ2(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgelq2_")]
        public static extern void ZGELQ2(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgelqf_")]
        public static extern void CGELQF(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgelqf_")]
        public static extern void DGELQF(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgelqf_")]
        public static extern void SGELQF(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgelqf_")]
        public static extern void ZGELQF(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgels_")]
        public static extern void CGELS(
         ref char trans,
         ref int m, ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgels_")]
        public static extern void DGELS(
         ref char trans,
         ref int m, ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgels_")]
        public static extern void SGELS(
         ref char trans,
         ref int m, ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgels_")]
        public static extern void ZGELS(
         ref char trans,
         ref int m, ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgelsd_")]
        public static extern void CGELSD(
         ref int m, ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref float S,
         ref float rcond, ref int rank,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgelsd_")]
        public static extern void DGELSD(
         ref int m, ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double S,
         ref double rcond, ref int rank,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgelsd_")]
        public static extern void SGELSD(
         ref int m, ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float S,
         ref float rcond, ref int rank,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgelsd_")]
        public static extern void ZGELSD(
         ref int m, ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref double S,
         ref double rcond, ref int rank,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgelss_")]
        public static extern void CGELSS(
         ref int m, ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref float S,
         ref float rcond, ref int rank,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgelss_")]
        public static extern void DGELSS(
         ref int m, ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double S,
         ref double rcond, ref int rank,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgelss_")]
        public static extern void SGELSS(
         ref int m, ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float S,
         ref float rcond, ref int rank,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgelss_")]
        public static extern void ZGELSS(
         ref int m, ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref double S,
         ref double rcond, ref int rank,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgelsy_")]
        public static extern void CGELSY(
         ref int m, ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb, ref int JPVT,
         ref float rcond, ref int rank,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgelsy_")]
        public static extern void DGELSY(
         ref int m, ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb, ref int JPVT,
         ref double rcond, ref int rank,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgelsy_")]
        public static extern void SGELSY(
         ref int m, ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float B, ref int ldb, ref int JPVT,
         ref float rcond, ref int rank,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgelsy_")]
        public static extern void ZGELSY(
         ref int m, ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb, ref int JPVT,
         ref double rcond, ref int rank,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgemlq_")]
        public static extern void CGEMLQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float T, ref int tsize,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgemlq_")]
        public static extern void DGEMLQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double T, ref int tsize,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgemlq_")]
        public static extern void SGEMLQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float T, ref int tsize,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgemlq_")]
        public static extern void ZGEMLQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double T, ref int tsize,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgemqr_")]
        public static extern void CGEMQR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float T, ref int tsize,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgemqr_")]
        public static extern void DGEMQR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double T, ref int tsize,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgemqr_")]
        public static extern void SGEMQR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float T, ref int tsize,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgemqr_")]
        public static extern void ZGEMQR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double T, ref int tsize,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgemqrt_")]
        public static extern void CGEMQRT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int nb,
         ref complex_float V, ref int ldv,
         ref complex_float T, ref int ldt,
         ref complex_float C, ref int ldc,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgemqrt_")]
        public static extern void DGEMQRT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int nb,
         ref double V, ref int ldv,
         ref double T, ref int ldt,
         ref double C, ref int ldc,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgemqrt_")]
        public static extern void SGEMQRT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int nb,
         ref float V, ref int ldv,
         ref float T, ref int ldt,
         ref float C, ref int ldc,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgemqrt_")]
        public static extern void ZGEMQRT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int nb,
         ref complex_double V, ref int ldv,
         ref complex_double T, ref int ldt,
         ref complex_double C, ref int ldc,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeql2_")]
        public static extern void CGEQL2(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeql2_")]
        public static extern void DGEQL2(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeql2_")]
        public static extern void SGEQL2(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeql2_")]
        public static extern void ZGEQL2(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqlf_")]
        public static extern void CGEQLF(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqlf_")]
        public static extern void DGEQLF(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqlf_")]
        public static extern void SGEQLF(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqlf_")]
        public static extern void ZGEQLF(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqpf_")]
        public static extern void SGEQPF(ref int m, ref int n, ref float a, ref int lda,
         ref int jpvt, ref float tau, ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqpf_")]
        public static extern void DGEQPF(ref int m, ref int n, ref double a, ref int lda,
         ref int jpvt, ref double tau, ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqpf_")]
        public static extern void CGEQPF(ref int m, ref int n, ref complex_float a,
         ref int lda, ref int jpvt,
         ref complex_float tau, ref complex_float work,
         ref float rwork, ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqpf_")]
        public static extern void ZGEQPF(ref int m, ref int n, ref complex_double a,
         ref int lda, ref int jpvt,
         ref complex_double tau, ref complex_double work,
         ref double rwork, ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqp3_")]
        public static extern void CGEQP3(
         ref int m, ref int n,
         ref complex_float A, ref int lda, ref int JPVT,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqp3_")]
        public static extern void DGEQP3(
         ref int m, ref int n,
         ref double A, ref int lda, ref int JPVT,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqp3_")]
        public static extern void SGEQP3(
         ref int m, ref int n,
         ref float A, ref int lda, ref int JPVT,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqp3_")]
        public static extern void ZGEQP3(
         ref int m, ref int n,
         ref complex_double A, ref int lda, ref int JPVT,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqr_")]
        public static extern void CGEQR(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float T, ref int tsize,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqr_")]
        public static extern void DGEQR(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double T, ref int tsize,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqr_")]
        public static extern void SGEQR(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float T, ref int tsize,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqr_")]
        public static extern void ZGEQR(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double T, ref int tsize,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqr2_")]
        public static extern void CGEQR2(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqr2_")]
        public static extern void DGEQR2(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqr2_")]
        public static extern void SGEQR2(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqr2_")]
        public static extern void ZGEQR2(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqrf_")]
        public static extern void CGEQRF(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqrf_")]
        public static extern void DGEQRF(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqrf_")]
        public static extern void SGEQRF(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqrf_")]
        public static extern void ZGEQRF(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqrfp_")]
        public static extern void CGEQRFP(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqrfp_")]
        public static extern void DGEQRFP(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqrfp_")]
        public static extern void SGEQRFP(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqrfp_")]
        public static extern void ZGEQRFP(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqrt_")]
        public static extern void CGEQRT(
         ref int m, ref int n, ref int nb,
         ref complex_float A, ref int lda,
         ref complex_float T, ref int ldt,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqrt_")]
        public static extern void DGEQRT(
         ref int m, ref int n, ref int nb,
         ref double A, ref int lda,
         ref double T, ref int ldt,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqrt_")]
        public static extern void SGEQRT(
         ref int m, ref int n, ref int nb,
         ref float A, ref int lda,
         ref float T, ref int ldt,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqrt_")]
        public static extern void ZGEQRT(
         ref int m, ref int n, ref int nb,
         ref complex_double A, ref int lda,
         ref complex_double T, ref int ldt,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqrt2_")]
        public static extern void CGEQRT2(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqrt2_")]
        public static extern void DGEQRT2(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqrt2_")]
        public static extern void SGEQRT2(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqrt2_")]
        public static extern void ZGEQRT2(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgeqrt3_")]
        public static extern void CGEQRT3(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgeqrt3_")]
        public static extern void DGEQRT3(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgeqrt3_")]
        public static extern void SGEQRT3(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgeqrt3_")]
        public static extern void ZGEQRT3(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgerfs_")]
        public static extern void CGERFS(
         ref char trans,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgerfs_")]
        public static extern void DGERFS(
         ref char trans,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref int ipiv,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgerfs_")]
        public static extern void SGERFS(
         ref char trans,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref int ipiv,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgerfs_")]
        public static extern void ZGERFS(
         ref char trans,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgerfsx_")]
        public static extern void CGERFSX(
         ref char trans, ref char equed,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv,
         ref float R,
         ref float C,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgerfsx_")]
        public static extern void DGERFSX(
         ref char trans, ref char equed,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref int ipiv,
         ref double R,
         ref double C,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgerfsx_")]
        public static extern void SGERFSX(
         ref char trans, ref char equed,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref int ipiv,
         ref float R,
         ref float C,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgerfsx_")]
        public static extern void ZGERFSX(
         ref char trans, ref char equed,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv,
         ref double R,
         ref double C,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgerq2_")]
        public static extern void CGERQ2(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgerq2_")]
        public static extern void DGERQ2(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgerq2_")]
        public static extern void SGERQ2(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgerq2_")]
        public static extern void ZGERQ2(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgerqf_")]
        public static extern void CGERQF(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgerqf_")]
        public static extern void DGERQF(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgerqf_")]
        public static extern void SGERQF(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgerqf_")]
        public static extern void ZGERQF(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgesdd_")]
        public static extern void CGESDD(
         ref char jobz,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float S,
         ref complex_float U, ref int ldu,
         ref complex_float VT, ref int ldvt,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgesdd_")]
        public static extern void DGESDD(
         ref char jobz,
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double S,
         ref double U, ref int ldu,
         ref double VT, ref int ldvt,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgesdd_")]
        public static extern void SGESDD(
         ref char jobz,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float S,
         ref float U, ref int ldu,
         ref float VT, ref int ldvt,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgesdd_")]
        public static extern void ZGESDD(
         ref char jobz,
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double S,
         ref complex_double U, ref int ldu,
         ref complex_double VT, ref int ldvt,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgesv_")]
        public static extern void CGESV(
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgesv_")]
        public static extern void DGESV(
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgesv_")]
        public static extern void SGESV(
         ref int n, ref int nrhs,
         ref float A, ref int lda, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgesv_")]
        public static extern void ZGESV(
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsgesv_")]
        public static extern void DSGESV(
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double work,
         ref float swork, ref int iter,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zcgesv_")]
        public static extern void ZCGESV(
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref complex_double work,
         ref complex_float swork,
         ref double rwork, ref int iter,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgesvd_")]
        public static extern void CGESVD(
         ref char jobu, ref char jobvt,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float S,
         ref complex_float U, ref int ldu,
         ref complex_float VT, ref int ldvt,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgesvd_")]
        public static extern void DGESVD(
         ref char jobu, ref char jobvt,
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double S,
         ref double U, ref int ldu,
         ref double VT, ref int ldvt,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgesvd_")]
        public static extern void SGESVD(
         ref char jobu, ref char jobvt,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float S,
         ref float U, ref int ldu,
         ref float VT, ref int ldvt,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgesvd_")]
        public static extern void ZGESVD(
         ref char jobu, ref char jobvt,
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double S,
         ref complex_double U, ref int ldu,
         ref complex_double VT, ref int ldvt,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgesvdq_")]
        public static extern void CGESVDQ(
         ref char joba, ref char jobp, ref char jobr, ref char jobu, ref char jobv,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float S,
         ref complex_float U, ref int ldu,
         ref complex_float V, ref int ldv, ref int numrank,
         ref int iwork, ref int liwork,
         ref complex_float cwork, ref int lcwork,
         ref float rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgesvdq_")]
        public static extern void DGESVDQ(
         ref char joba, ref char jobp, ref char jobr, ref char jobu, ref char jobv,
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double S,
         ref double U, ref int ldu,
         ref double V, ref int ldv, ref int numrank,
         ref int iwork, ref int liwork,
         ref double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgesvdq_")]
        public static extern void SGESVDQ(
         ref char joba, ref char jobp, ref char jobr, ref char jobu, ref char jobv,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float S,
         ref float U, ref int ldu,
         ref float V, ref int ldv, ref int numrank,
         ref int iwork, ref int liwork,
         ref float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgesvdq_")]
        public static extern void ZGESVDQ(
         ref char joba, ref char jobp, ref char jobr, ref char jobu, ref char jobv,
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double S,
         ref complex_double U, ref int ldu,
         ref complex_double V, ref int ldv, ref int numrank,
         ref int iwork, ref int liwork,
         ref complex_float cwork, ref int lcwork,
         ref double rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgesvdx_")]
        public static extern void CGESVDX(
         ref char jobu, ref char jobvt, ref char range,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu, ref int ns,
         ref float S,
         ref complex_float U, ref int ldu,
         ref complex_float VT, ref int ldvt,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgesvdx_")]
        public static extern void DGESVDX(
         ref char jobu, ref char jobvt, ref char range,
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu, ref int ns,
         ref double S,
         ref double U, ref int ldu,
         ref double VT, ref int ldvt,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgesvdx_")]
        public static extern void SGESVDX(
         ref char jobu, ref char jobvt, ref char range,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu, ref int ns,
         ref float S,
         ref float U, ref int ldu,
         ref float VT, ref int ldvt,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgesvdx_")]
        public static extern void ZGESVDX(
         ref char jobu, ref char jobvt, ref char range,
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu, ref int ns,
         ref double S,
         ref complex_double U, ref int ldu,
         ref complex_double VT, ref int ldvt,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgesvj_")]
        public static extern void CGESVJ(
         ref char joba, ref char jobu, ref char jobv,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float SVA, ref int mv,
         ref complex_float V, ref int ldv,
         ref complex_float cwork, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgesvj_")]
        public static extern void DGESVJ(
         ref char joba, ref char jobu, ref char jobv,
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double SVA, ref int mv,
         ref double V, ref int ldv,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgesvj_")]
        public static extern void SGESVJ(
         ref char joba, ref char jobu, ref char jobv,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float SVA, ref int mv,
         ref float V, ref int ldv,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgesvj_")]
        public static extern void ZGESVJ(
         ref char joba, ref char jobu, ref char jobv,
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double SVA, ref int mv,
         ref complex_double V, ref int ldv,
         ref complex_double cwork, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgesvx_")]
        public static extern void CGESVX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv, ref char equed,
         ref float R,
         ref float C,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgesvx_")]
        public static extern void DGESVX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref int ipiv, ref char equed,
         ref double R,
         ref double C,
         ref double B,
         ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgesvx_")]
        public static extern void SGESVX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref int ipiv, ref char equed,
         ref float R,
         ref float C,
         ref float B,
         ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgesvx_")]
        public static extern void ZGESVX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv, ref char equed,
         ref double R,
         ref double C,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgesvxx_")]
        public static extern void CGESVXX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv, ref char equed,
         ref float R,
         ref float C,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float rpvgrw,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgesvxx_")]
        public static extern void DGESVXX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref int ipiv, ref char equed,
         ref double R,
         ref double C,
         ref double B,
         ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double rpvgrw,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgesvxx_")]
        public static extern void SGESVXX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref int ipiv, ref char equed,
         ref float R,
         ref float C,
         ref float B,
         ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float rpvgrw,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgesvxx_")]
        public static extern void ZGESVXX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv, ref char equed,
         ref double R,
         ref double C,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double rpvgrw,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgetf2_")]
        public static extern void CGETF2(
         ref int m, ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgetf2_")]
        public static extern void DGETF2(
         ref int m, ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgetf2_")]
        public static extern void SGETF2(
         ref int m, ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgetf2_")]
        public static extern void ZGETF2(
         ref int m, ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgetrf_")]
        public static extern void CGETRF(
         ref int m, ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgetrf_")]
        public static extern void DGETRF(
         ref int m, ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgetrf_")]
        public static extern void SGETRF(
         ref int m, ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgetrf_")]
        public static extern void ZGETRF(
         ref int m, ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgetrf2_")]
        public static extern void CGETRF2(
         ref int m, ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgetrf2_")]
        public static extern void DGETRF2(
         ref int m, ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgetrf2_")]
        public static extern void SGETRF2(
         ref int m, ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgetrf2_")]
        public static extern void ZGETRF2(
         ref int m, ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgetri_")]
        public static extern void CGETRI(
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgetri_")]
        public static extern void DGETRI(
         ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgetri_")]
        public static extern void SGETRI(
         ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgetri_")]
        public static extern void ZGETRI(
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgetrs_")]
        public static extern void CGETRS(
         ref char trans,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgetrs_")]
        public static extern void DGETRS(
         ref char trans,
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgetrs_")]
        public static extern void SGETRS(
         ref char trans,
         ref int n, ref int nrhs,
         ref float A, ref int lda, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgetrs_")]
        public static extern void ZGETRS(
         ref char trans,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgetsls_")]
        public static extern void CGETSLS(
         ref char trans,
         ref int m, ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgetsls_")]
        public static extern void DGETSLS(
         ref char trans,
         ref int m, ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgetsls_")]
        public static extern void SGETSLS(
         ref char trans,
         ref int m, ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgetsls_")]
        public static extern void ZGETSLS(
         ref char trans,
         ref int m, ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggbak_")]
        public static extern void CGGBAK(
         ref char job, ref char side,
         ref int n, ref int ilo, ref int ihi,
         ref float lscale,
         ref float rscale, ref int m,
         ref complex_float V, ref int ldv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggbak_")]
        public static extern void DGGBAK(
         ref char job, ref char side,
         ref int n, ref int ilo, ref int ihi,
         ref double lscale,
         ref double rscale, ref int m,
         ref double V, ref int ldv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggbak_")]
        public static extern void SGGBAK(
         ref char job, ref char side,
         ref int n, ref int ilo, ref int ihi,
         ref float lscale,
         ref float rscale, ref int m,
         ref float V, ref int ldv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggbak_")]
        public static extern void ZGGBAK(
         ref char job, ref char side,
         ref int n, ref int ilo, ref int ihi,
         ref double lscale,
         ref double rscale, ref int m,
         ref complex_double V, ref int ldv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggbal_")]
        public static extern void CGGBAL(
         ref char job,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb, ref int ilo, ref int ihi,
         ref float lscale,
         ref float rscale,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggbal_")]
        public static extern void DGGBAL(
         ref char job,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb, ref int ilo, ref int ihi,
         ref double lscale,
         ref double rscale,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggbal_")]
        public static extern void SGGBAL(
         ref char job,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb, ref int ilo, ref int ihi,
         ref float lscale,
         ref float rscale,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggbal_")]
        public static extern void ZGGBAL(
         ref char job,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb, ref int ilo, ref int ihi,
         ref double lscale,
         ref double rscale,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgges_")]
        public static extern void CGGES(
         ref char jobvsl, ref char jobvsr, ref char sort, ref C_SELECT2 selctg,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb, ref int sdim,
         ref complex_float alpha,
         ref complex_float beta,
         ref complex_float VSL, ref int ldvsl,
         ref complex_float VSR, ref int ldvsr,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgges_")]
        public static extern void DGGES(
         ref char jobvsl, ref char jobvsr, ref char sort, ref D_SELECT3 selctg,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb, ref int sdim,
         ref double alphar,
         ref double alphai,
         ref double beta,
         ref double VSL, ref int ldvsl,
         ref double VSR, ref int ldvsr,
         ref double work, ref int lwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgges_")]
        public static extern void SGGES(
         ref char jobvsl, ref char jobvsr, ref char sort, ref S_SELECT3 selctg,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb, ref int sdim,
         ref float alphar,
         ref float alphai,
         ref float beta,
         ref float VSL, ref int ldvsl,
         ref float VSR, ref int ldvsr,
         ref float work, ref int lwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgges_")]
        public static extern void ZGGES(
         ref char jobvsl, ref char jobvsr, ref char sort, ref Z_SELECT2 selctg,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb, ref int sdim,
         ref complex_double alpha,
         ref complex_double beta,
         ref complex_double VSL, ref int ldvsl,
         ref complex_double VSR, ref int ldvsr,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgges3_")]
        public static extern void CGGES3(
         ref char jobvsl, ref char jobvsr, ref char sort, ref C_SELECT2 selctg,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb, ref int sdim,
         ref complex_float alpha,
         ref complex_float beta,
         ref complex_float VSL, ref int ldvsl,
         ref complex_float VSR, ref int ldvsr,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgges3_")]
        public static extern void DGGES3(
         ref char jobvsl, ref char jobvsr, ref char sort, ref D_SELECT2 selctg,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb, ref int sdim,
         ref double alphar,
         ref double alphai,
         ref double beta,
         ref double VSL, ref int ldvsl,
         ref double VSR, ref int ldvsr,
         ref double work, ref int lwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgges3_")]
        public static extern void SGGES3(
         ref char jobvsl, ref char jobvsr, ref char sort, ref S_SELECT2 selctg,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb, ref int sdim,
         ref float alphar,
         ref float alphai,
         ref float beta,
         ref float VSL, ref int ldvsl,
         ref float VSR, ref int ldvsr,
         ref float work, ref int lwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgges3_")]
        public static extern void ZGGES3(
         ref char jobvsl, ref char jobvsr, ref char sort, ref Z_SELECT2 selctg,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb, ref int sdim,
         ref complex_double alpha,
         ref complex_double beta,
         ref complex_double VSL, ref int ldvsl,
         ref complex_double VSR, ref int ldvsr,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggesx_")]
        public static extern void CGGESX(
         ref char jobvsl, ref char jobvsr, ref char sort, ref C_SELECT2 selctg, ref char sense,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb, ref int sdim,
         ref complex_float alpha,
         ref complex_float beta,
         ref complex_float VSL, ref int ldvsl,
         ref complex_float VSR, ref int ldvsr,
         ref float rconde,
         ref float rcondv,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork, ref int liwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggesx_")]
        public static extern void DGGESX(
         ref char jobvsl, ref char jobvsr, ref char sort, ref D_SELECT2 selctg, ref char sense,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb, ref int sdim,
         ref double alphar,
         ref double alphai,
         ref double beta,
         ref double VSL, ref int ldvsl,
         ref double VSR, ref int ldvsr,
         ref double rconde,
         ref double rcondv,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggesx_")]
        public static extern void SGGESX(
         ref char jobvsl, ref char jobvsr, ref char sort, ref S_SELECT2 selctg, ref char sense,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb, ref int sdim,
         ref float alphar,
         ref float alphai,
         ref float beta,
         ref float VSL, ref int ldvsl,
         ref float VSR, ref int ldvsr,
         ref float rconde,
         ref float rcondv,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggesx_")]
        public static extern void ZGGESX(
         ref char jobvsl, ref char jobvsr, ref char sort, ref Z_SELECT2 selctg, ref char sense,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb, ref int sdim,
         ref complex_double alpha,
         ref complex_double beta,
         ref complex_double VSL, ref int ldvsl,
         ref complex_double VSR, ref int ldvsr,
         ref double rconde,
         ref double rcondv,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork, ref int liwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggev_")]
        public static extern void CGGEV(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float alpha,
         ref complex_float beta,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggev_")]
        public static extern void DGGEV(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double alphar,
         ref double alphai,
         ref double beta,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggev_")]
        public static extern void SGGEV(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float alphar,
         ref float alphai,
         ref float beta,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggev_")]
        public static extern void ZGGEV(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double alpha,
         ref complex_double beta,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggev3_")]
        public static extern void CGGEV3(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float alpha,
         ref complex_float beta,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggev3_")]
        public static extern void DGGEV3(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double alphar,
         ref double alphai,
         ref double beta,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggev3_")]
        public static extern void SGGEV3(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float alphar,
         ref float alphai,
         ref float beta,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggev3_")]
        public static extern void ZGGEV3(
         ref char jobvl, ref char jobvr,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double alpha,
         ref complex_double beta,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggevx_")]
        public static extern void CGGEVX(
         ref char balanc, ref char jobvl, ref char jobvr, ref char sense,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float alpha,
         ref complex_float beta,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr, ref int ilo, ref int ihi,
         ref float lscale,
         ref float rscale,
         ref float abnrm,
         ref float bbnrm,
         ref float rconde,
         ref float rcondv,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggevx_")]
        public static extern void DGGEVX(
         ref char balanc, ref char jobvl, ref char jobvr, ref char sense,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double alphar,
         ref double alphai,
         ref double beta,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr, ref int ilo, ref int ihi,
         ref double lscale,
         ref double rscale,
         ref double abnrm,
         ref double bbnrm,
         ref double rconde,
         ref double rcondv,
         ref double work, ref int lwork,
         ref int iwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggevx_")]
        public static extern void SGGEVX(
         ref char balanc, ref char jobvl, ref char jobvr, ref char sense,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float alphar,
         ref float alphai,
         ref float beta,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr, ref int ilo, ref int ihi,
         ref float lscale,
         ref float rscale,
         ref float abnrm,
         ref float bbnrm,
         ref float rconde,
         ref float rcondv,
         ref float work, ref int lwork,
         ref int iwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggevx_")]
        public static extern void ZGGEVX(
         ref char balanc, ref char jobvl, ref char jobvr, ref char sense,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double alpha,
         ref complex_double beta,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr, ref int ilo, ref int ihi,
         ref double lscale,
         ref double rscale,
         ref double abnrm,
         ref double bbnrm,
         ref double rconde,
         ref double rcondv,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork, ref int BWORK,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggglm_")]
        public static extern void CGGGLM(
         ref int n, ref int m, ref int p,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float D,
         ref complex_float X,
         ref complex_float Y,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggglm_")]
        public static extern void DGGGLM(
         ref int n, ref int m, ref int p,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double D,
         ref double X,
         ref double Y,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggglm_")]
        public static extern void SGGGLM(
         ref int n, ref int m, ref int p,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float D,
         ref float X,
         ref float Y,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggglm_")]
        public static extern void ZGGGLM(
         ref int n, ref int m, ref int p,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double D,
         ref complex_double X,
         ref complex_double Y,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgghd3_")]
        public static extern void CGGHD3(
         ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float Q, ref int ldq,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgghd3_")]
        public static extern void DGGHD3(
         ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double Q, ref int ldq,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgghd3_")]
        public static extern void SGGHD3(
         ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float Q, ref int ldq,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgghd3_")]
        public static extern void ZGGHD3(
         ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double Q, ref int ldq,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgghrd_")]
        public static extern void CGGHRD(
         ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float Q, ref int ldq,
         ref complex_float Z, ref int ldz,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgghrd_")]
        public static extern void DGGHRD(
         ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double Q, ref int ldq,
         ref double Z, ref int ldz,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgghrd_")]
        public static extern void SGGHRD(
         ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float Q, ref int ldq,
         ref float Z, ref int ldz,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgghrd_")]
        public static extern void ZGGHRD(
         ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double Q, ref int ldq,
         ref complex_double Z, ref int ldz,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgglse_")]
        public static extern void CGGLSE(
         ref int m, ref int n, ref int p,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float C,
         ref complex_float D,
         ref complex_float X,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgglse_")]
        public static extern void DGGLSE(
         ref int m, ref int n, ref int p,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double C,
         ref double D,
         ref double X,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgglse_")]
        public static extern void SGGLSE(
         ref int m, ref int n, ref int p,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float C,
         ref float D,
         ref float X,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgglse_")]
        public static extern void ZGGLSE(
         ref int m, ref int n, ref int p,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double C,
         ref complex_double D,
         ref complex_double X,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggqrf_")]
        public static extern void CGGQRF(
         ref int n, ref int m, ref int p,
         ref complex_float A, ref int lda,
         ref complex_float taua,
         ref complex_float B, ref int ldb,
         ref complex_float taub,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggqrf_")]
        public static extern void DGGQRF(
         ref int n, ref int m, ref int p,
         ref double A, ref int lda,
         ref double taua,
         ref double B, ref int ldb,
         ref double taub,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggqrf_")]
        public static extern void SGGQRF(
         ref int n, ref int m, ref int p,
         ref float A, ref int lda,
         ref float taua,
         ref float B, ref int ldb,
         ref float taub,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggqrf_")]
        public static extern void ZGGQRF(
         ref int n, ref int m, ref int p,
         ref complex_double A, ref int lda,
         ref complex_double taua,
         ref complex_double B, ref int ldb,
         ref complex_double taub,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggrqf_")]
        public static extern void CGGRQF(
         ref int m, ref int p, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float taua,
         ref complex_float B, ref int ldb,
         ref complex_float taub,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggrqf_")]
        public static extern void DGGRQF(
         ref int m, ref int p, ref int n,
         ref double A, ref int lda,
         ref double taua,
         ref double B, ref int ldb,
         ref double taub,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggrqf_")]
        public static extern void SGGRQF(
         ref int m, ref int p, ref int n,
         ref float A, ref int lda,
         ref float taua,
         ref float B, ref int ldb,
         ref float taub,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggrqf_")]
        public static extern void ZGGRQF(
         ref int m, ref int p, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double taua,
         ref complex_double B, ref int ldb,
         ref complex_double taub,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggsvd_")]
        public static extern int SGGSVD(int matrix_layout, ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int n, ref int p,
         ref int k, ref int l, ref float a,
         ref int lda, ref float b, ref int ldb,
         ref float alpha, ref float beta, ref float u, ref int ldu,
         ref float v, ref int ldv, ref float q, ref int ldq,
         ref int iwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggsvd_")]
        public static extern int DGGSVD(int matrix_layout, ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int n, ref int p,
         ref int k, ref int l, ref double a,
         ref int lda, ref double b, ref int ldb,
         ref double alpha, ref double beta, ref double u,
         ref int ldu, ref double v, ref int ldv, ref double q,
         ref int ldq, ref int iwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggsvd_")]
        public static extern int CGGSVD(int matrix_layout, ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int n, ref int p,
         ref int k, ref int l,
         ref complex_float a, ref int lda,
         ref complex_float b, ref int ldb,
         ref float alpha, ref float beta, ref complex_float u,
         ref int ldu, ref complex_float v,
         ref int ldv, ref complex_float q,
         ref int ldq, ref int iwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggsvd_")]
        public static extern int ZGGSVD(int matrix_layout, ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int n, ref int p,
         ref int k, ref int l,
         ref complex_double a, ref int lda,
         ref complex_double b, ref int ldb,
         ref double alpha, ref double beta,
         ref complex_double u, ref int ldu,
         ref complex_double v, ref int ldv,
         ref complex_double q, ref int ldq,
         ref int iwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggsvd3_")]
        public static extern void CGGSVD3(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int n, ref int p, ref int k, ref int l,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref float alpha,
         ref float beta,
         ref complex_float U, ref int ldu,
         ref complex_float V, ref int ldv,
         ref complex_float Q, ref int ldq,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggsvd3_")]
        public static extern void DGGSVD3(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int n, ref int p, ref int k, ref int l,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double alpha,
         ref double beta,
         ref double U, ref int ldu,
         ref double V, ref int ldv,
         ref double Q, ref int ldq,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggsvd3_")]
        public static extern void SGGSVD3(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int n, ref int p, ref int k, ref int l,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float alpha,
         ref float beta,
         ref float U, ref int ldu,
         ref float V, ref int ldv,
         ref float Q, ref int ldq,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggsvd3_")]
        public static extern void ZGGSVD3(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int n, ref int p, ref int k, ref int l,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref double alpha,
         ref double beta,
         ref complex_double U, ref int ldu,
         ref complex_double V, ref int ldv,
         ref complex_double Q, ref int ldq,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggsvp_")]
        public static extern int SGGSVP(int matrix_layout, ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n, ref float a,
         ref int lda, ref float b, ref int ldb, ref float tola,
         ref float tolb, ref int k, ref int l, ref float u,
         ref int ldu, ref float v, ref int ldv, ref float q,
         ref int ldq);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggsvp_")]
        public static extern int DGGSVP(int matrix_layout, ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n, ref double a,
         ref int lda, ref double b, ref int ldb,
         ref double tola, ref double tolb, ref int k,
         ref int l, ref double u, ref int ldu, ref double v,
         ref int ldv, ref double q, ref int ldq);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggsvp_")]
        public static extern int CGGSVP(int matrix_layout, ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n,
         ref complex_float a, ref int lda,
         ref complex_float b, ref int ldb, ref float tola,
         ref float tolb, ref int k, ref int l,
         ref complex_float u, ref int ldu,
         ref complex_float v, ref int ldv,
         ref complex_float q, ref int ldq);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggsvp_")]
        public static extern int ZGGSVP(int matrix_layout, ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n,
         ref complex_double a, ref int lda,
         ref complex_double b, ref int ldb,
         ref double tola, ref double tolb, ref int k,
         ref int l, ref complex_double u,
         ref int ldu, ref complex_double v,
         ref int ldv, ref complex_double q,
         ref int ldq);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cggsvp3_")]
        public static extern void CGGSVP3(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref float tola,
         ref float tolb, ref int k, ref int l,
         ref complex_float U, ref int ldu,
         ref complex_float V, ref int ldv,
         ref complex_float Q, ref int ldq,
         ref int iwork,
         ref float rwork,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dggsvp3_")]
        public static extern void DGGSVP3(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double tola,
         ref double tolb, ref int k, ref int l,
         ref double U, ref int ldu,
         ref double V, ref int ldv,
         ref double Q, ref int ldq,
         ref int iwork,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sggsvp3_")]
        public static extern void SGGSVP3(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float tola,
         ref float tolb, ref int k, ref int l,
         ref float U, ref int ldu,
         ref float V, ref int ldv,
         ref float Q, ref int ldq,
         ref int iwork,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zggsvp3_")]
        public static extern void ZGGSVP3(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref double tola,
         ref double tolb, ref int k, ref int l,
         ref complex_double U, ref int ldu,
         ref complex_double V, ref int ldv,
         ref complex_double Q, ref int ldq,
         ref int iwork,
         ref double rwork,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgtcon_")]
        public static extern void CGTCON(
         ref char norm,
         ref int n,
         ref complex_float DL,
         ref complex_float D,
         ref complex_float DU,
         ref complex_float DU2, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgtcon_")]
        public static extern void DGTCON(
         ref char norm,
         ref int n,
         ref double DL,
         ref double D,
         ref double DU,
         ref double DU2, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgtcon_")]
        public static extern void SGTCON(
         ref char norm,
         ref int n,
         ref float DL,
         ref float D,
         ref float DU,
         ref float DU2, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgtcon_")]
        public static extern void ZGTCON(
         ref char norm,
         ref int n,
         ref complex_double DL,
         ref complex_double D,
         ref complex_double DU,
         ref complex_double DU2, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgtrfs_")]
        public static extern void CGTRFS(
         ref char trans,
         ref int n, ref int nrhs,
         ref complex_float DL,
         ref complex_float D,
         ref complex_float DU,
         ref complex_float DLF,
         ref complex_float DF,
         ref complex_float DUF,
         ref complex_float DU2, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgtrfs_")]
        public static extern void DGTRFS(
         ref char trans,
         ref int n, ref int nrhs,
         ref double DL,
         ref double D,
         ref double DU,
         ref double DLF,
         ref double DF,
         ref double DUF,
         ref double DU2, ref int ipiv,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgtrfs_")]
        public static extern void SGTRFS(
         ref char trans,
         ref int n, ref int nrhs,
         ref float DL,
         ref float D,
         ref float DU,
         ref float DLF,
         ref float DF,
         ref float DUF,
         ref float DU2, ref int ipiv,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgtrfs_")]
        public static extern void ZGTRFS(
         ref char trans,
         ref int n, ref int nrhs,
         ref complex_double DL,
         ref complex_double D,
         ref complex_double DU,
         ref complex_double DLF,
         ref complex_double DF,
         ref complex_double DUF,
         ref complex_double DU2, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgtsv_")]
        public static extern void CGTSV(
         ref int n, ref int nrhs,
         ref complex_float DL,
         ref complex_float D,
         ref complex_float DU,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgtsv_")]
        public static extern void DGTSV(
         ref int n, ref int nrhs,
         ref double DL,
         ref double D,
         ref double DU,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgtsv_")]
        public static extern void SGTSV(
         ref int n, ref int nrhs,
         ref float DL,
         ref float D,
         ref float DU,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgtsv_")]
        public static extern void ZGTSV(
         ref int n, ref int nrhs,
         ref complex_double DL,
         ref complex_double D,
         ref complex_double DU,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgtsvx_")]
        public static extern void CGTSVX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref complex_float DL,
         ref complex_float D,
         ref complex_float DU,
         ref complex_float DLF,
         ref complex_float DF,
         ref complex_float DUF,
         ref complex_float DU2, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgtsvx_")]
        public static extern void DGTSVX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref double DL,
         ref double D,
         ref double DU,
         ref double DLF,
         ref double DF,
         ref double DUF,
         ref double DU2, ref int ipiv,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgtsvx_")]
        public static extern void SGTSVX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref float DL,
         ref float D,
         ref float DU,
         ref float DLF,
         ref float DF,
         ref float DUF,
         ref float DU2, ref int ipiv,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgtsvx_")]
        public static extern void ZGTSVX(
         ref char fact, ref char trans,
         ref int n, ref int nrhs,
         ref complex_double DL,
         ref complex_double D,
         ref complex_double DU,
         ref complex_double DLF,
         ref complex_double DF,
         ref complex_double DUF,
         ref complex_double DU2, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgttrf_")]
        public static extern void CGTTRF(
         ref int n,
         ref complex_float DL,
         ref complex_float D,
         ref complex_float DU,
         ref complex_float DU2, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgttrf_")]
        public static extern void DGTTRF(
         ref int n,
         ref double DL,
         ref double D,
         ref double DU,
         ref double DU2, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgttrf_")]
        public static extern void SGTTRF(
         ref int n,
         ref float DL,
         ref float D,
         ref float DU,
         ref float DU2, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgttrf_")]
        public static extern void ZGTTRF(
         ref int n,
         ref complex_double DL,
         ref complex_double D,
         ref complex_double DU,
         ref complex_double DU2, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cgttrs_")]
        public static extern void CGTTRS(
         ref char trans,
         ref int n, ref int nrhs,
         ref complex_float DL,
         ref complex_float D,
         ref complex_float DU,
         ref complex_float DU2, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dgttrs_")]
        public static extern void DGTTRS(
         ref char trans,
         ref int n, ref int nrhs,
         ref double DL,
         ref double D,
         ref double DU,
         ref double DU2, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sgttrs_")]
        public static extern void SGTTRS(
         ref char trans,
         ref int n, ref int nrhs,
         ref float DL,
         ref float D,
         ref float DU,
         ref float DU2, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zgttrs_")]
        public static extern void ZGTTRS(
         ref char trans,
         ref int n, ref int nrhs,
         ref complex_double DL,
         ref complex_double D,
         ref complex_double DU,
         ref complex_double DU2, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbev_")]
        public static extern void CHBEV(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbev_")]
        public static extern void ZHBEV(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbev_2stage_")]
        public static extern void CHBEV_2STAGE(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbev_2stage_")]
        public static extern void ZHBEV_2STAGE(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbevd_")]
        public static extern void CHBEVD(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbevd_")]
        public static extern void ZHBEVD(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbevd_2stage_")]
        public static extern void CHBEVD_2STAGE(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbevd_2stage_")]
        public static extern void ZHBEVD_2STAGE(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbevx_")]
        public static extern void CHBEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref complex_float Q, ref int ldq,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work,
         ref float rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbevx_")]
        public static extern void ZHBEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref complex_double Q, ref int ldq,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work,
         ref double rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbevx_2stage_")]
        public static extern void CHBEVX_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref complex_float Q, ref int ldq,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbevx_2stage_")]
        public static extern void ZHBEVX_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref complex_double Q, ref int ldq,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbgst_")]
        public static extern void CHBGST(
         ref char vect, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref complex_float AB, ref int ldab,
         ref complex_float BB, ref int ldbb,
         ref complex_float X, ref int ldx,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbgst_")]
        public static extern void ZHBGST(
         ref char vect, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref complex_double AB, ref int ldab,
         ref complex_double BB, ref int ldbb,
         ref complex_double X, ref int ldx,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbgv_")]
        public static extern void CHBGV(
         ref char jobz, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref complex_float AB, ref int ldab,
         ref complex_float BB, ref int ldbb,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbgv_")]
        public static extern void ZHBGV(
         ref char jobz, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref complex_double AB, ref int ldab,
         ref complex_double BB, ref int ldbb,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbgvd_")]
        public static extern void CHBGVD(
         ref char jobz, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref complex_float AB, ref int ldab,
         ref complex_float BB, ref int ldbb,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbgvd_")]
        public static extern void ZHBGVD(
         ref char jobz, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref complex_double AB, ref int ldab,
         ref complex_double BB, ref int ldbb,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbgvx_")]
        public static extern void CHBGVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref complex_float AB, ref int ldab,
         ref complex_float BB, ref int ldbb,
         ref complex_float Q, ref int ldq,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work,
         ref float rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbgvx_")]
        public static extern void ZHBGVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref complex_double AB, ref int ldab,
         ref complex_double BB, ref int ldbb,
         ref complex_double Q, ref int ldq,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work,
         ref double rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chbtrd_")]
        public static extern void CHBTRD(
         ref char vect, ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref float D,
         ref float E,
         ref complex_float Q, ref int ldq,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhbtrd_")]
        public static extern void ZHBTRD(
         ref char vect, ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref double D,
         ref double E,
         ref complex_double Q, ref int ldq,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "checon_")]
        public static extern void CHECON(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhecon_")]
        public static extern void ZHECON(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "checon_3_")]
        public static extern void CHECON_3(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhecon_3_")]
        public static extern void ZHECON_3(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheequb_")]
        public static extern void CHEEQUB(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float S,
         ref float scond,
         ref float amax,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheequb_")]
        public static extern void ZHEEQUB(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double S,
         ref double scond,
         ref double amax,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheev_")]
        public static extern void CHEEV(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float W,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheev_")]
        public static extern void ZHEEV(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double W,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheev_2stage_")]
        public static extern void CHEEV_2STAGE(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float W,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheev_2stage_")]
        public static extern void ZHEEV_2STAGE(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double W,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheevd_")]
        public static extern void CHEEVD(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float W,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheevd_")]
        public static extern void ZHEEVD(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double W,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheevd_2stage_")]
        public static extern void CHEEVD_2STAGE(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float W,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheevd_2stage_")]
        public static extern void ZHEEVD_2STAGE(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double W,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheevr_")]
        public static extern void CHEEVR(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz, ref int ISUPPZ,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheevr_")]
        public static extern void ZHEEVR(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz, ref int ISUPPZ,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheevr_2stage_")]
        public static extern void CHEEVR_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz, ref int ISUPPZ,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheevr_2stage_")]
        public static extern void ZHEEVR_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz, ref int ISUPPZ,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheevx_")]
        public static extern void CHEEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheevx_")]
        public static extern void ZHEEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheevx_2stage_")]
        public static extern void CHEEVX_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheevx_2stage_")]
        public static extern void ZHEEVX_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chegst_")]
        public static extern void CHEGST(
         ref int itype, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhegst_")]
        public static extern void ZHEGST(
         ref int itype, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chegv_")]
        public static extern void CHEGV(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref float W,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhegv_")]
        public static extern void ZHEGV(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref double W,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chegv_2stage_")]
        public static extern void CHEGV_2STAGE(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref float W,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhegv_2stage_")]
        public static extern void ZHEGV_2STAGE(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref double W,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chegvd_")]
        public static extern void CHEGVD(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref float W,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhegvd_")]
        public static extern void ZHEGVD(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref double W,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chegvx_")]
        public static extern void CHEGVX(
         ref int itype, ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhegvx_")]
        public static extern void ZHEGVX(
         ref int itype, ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cherfs_")]
        public static extern void CHERFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zherfs_")]
        public static extern void ZHERFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cherfsx_")]
        public static extern void CHERFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv,
         ref float S,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zherfsx_")]
        public static extern void ZHERFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv,
         ref double S,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chesv_")]
        public static extern void CHESV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhesv_")]
        public static extern void ZHESV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chesv_aa_")]
        public static extern void CHESV_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhesv_aa_")]
        public static extern void ZHESV_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chesv_aa_2stage_")]
        public static extern void CHESV_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhesv_aa_2stage_")]
        public static extern void ZHESV_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chesv_rk_")]
        public static extern void CHESV_RK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhesv_rk_")]
        public static extern void ZHESV_RK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chesv_rook_")]
        public static extern void CHESV_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhesv_rook_")]
        public static extern void ZHESV_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chesvx_")]
        public static extern void CHESVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhesvx_")]
        public static extern void ZHESVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chesvxx_")]
        public static extern void CHESVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv, ref char equed,
         ref float S,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float rpvgrw,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhesvxx_")]
        public static extern void ZHESVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv, ref char equed,
         ref double S,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double rpvgrw,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cheswapr_")]
        public static extern void CHESWAPR(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int i1, ref int i2);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zheswapr_")]
        public static extern void ZHESWAPR(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int i1, ref int i2);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrd_")]
        public static extern void CHETRD(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float D,
         ref float E,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrd_")]
        public static extern void ZHETRD(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double D,
         ref double E,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrd_2stage_")]
        public static extern void CHETRD_2STAGE(
         ref char vect, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float D,
         ref float E,
         ref complex_float tau,
         ref complex_float HOUS2, ref int lhous2,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrd_2stage_")]
        public static extern void ZHETRD_2STAGE(
         ref char vect, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double D,
         ref double E,
         ref complex_double tau,
         ref complex_double HOUS2, ref int lhous2,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrf_")]
        public static extern void CHETRF(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrf_")]
        public static extern void ZHETRF(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrf_aa_")]
        public static extern void CHETRF_AA(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrf_aa_")]
        public static extern void ZHETRF_AA(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrf_aa_2stage_")]
        public static extern void CHETRF_AA_2STAGE(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrf_aa_2stage_")]
        public static extern void ZHETRF_AA_2STAGE(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrf_rk_")]
        public static extern void CHETRF_RK(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrf_rk_")]
        public static extern void ZHETRF_RK(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrf_rook_")]
        public static extern void CHETRF_ROOK(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrf_rook_")]
        public static extern void ZHETRF_ROOK(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetri_")]
        public static extern void CHETRI(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetri_")]
        public static extern void ZHETRI(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetri2_")]
        public static extern void CHETRI2(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetri2_")]
        public static extern void ZHETRI2(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetri2x_")]
        public static extern void CHETRI2X(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int nb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetri2x_")]
        public static extern void ZHETRI2X(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int nb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetri_3_")]
        public static extern void CHETRI_3(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetri_3_")]
        public static extern void ZHETRI_3(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrs_")]
        public static extern void CHETRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrs_")]
        public static extern void ZHETRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrs2_")]
        public static extern void CHETRS2(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrs2_")]
        public static extern void ZHETRS2(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrs_3_")]
        public static extern void CHETRS_3(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrs_3_")]
        public static extern void ZHETRS_3(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrs_aa_")]
        public static extern void CHETRS_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrs_aa_")]
        public static extern void ZHETRS_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrs_aa_2stage_")]
        public static extern void CHETRS_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrs_aa_2stage_")]
        public static extern void ZHETRS_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chetrs_rook_")]
        public static extern void CHETRS_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhetrs_rook_")]
        public static extern void ZHETRS_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chfrk_")]
        public static extern void CHFRK(
         ref char transr, ref char uplo, ref char trans,
         ref int n, ref int k,
         ref float alpha,
         ref complex_float A, ref int lda,
         ref float beta,
         ref complex_float C);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhfrk_")]
        public static extern void ZHFRK(
         ref char transr, ref char uplo, ref char trans,
         ref int n, ref int k,
         ref double alpha,
         ref complex_double A, ref int lda,
         ref double beta,
         ref complex_double C);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chgeqz_")]
        public static extern void CHGEQZ(
         ref char job, ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref complex_float H, ref int ldh,
         ref complex_float T, ref int ldt,
         ref complex_float alpha,
         ref complex_float beta,
         ref complex_float Q, ref int ldq,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dhgeqz_")]
        public static extern void DHGEQZ(
         ref char job, ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref double H, ref int ldh,
         ref double T, ref int ldt,
         ref double alphar,
         ref double alphai,
         ref double beta,
         ref double Q, ref int ldq,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "shgeqz_")]
        public static extern void SHGEQZ(
         ref char job, ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref float H, ref int ldh,
         ref float T, ref int ldt,
         ref float alphar,
         ref float alphai,
         ref float beta,
         ref float Q, ref int ldq,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhgeqz_")]
        public static extern void ZHGEQZ(
         ref char job, ref char compq, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref complex_double H, ref int ldh,
         ref complex_double T, ref int ldt,
         ref complex_double alpha,
         ref complex_double beta,
         ref complex_double Q, ref int ldq,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpcon_")]
        public static extern void CHPCON(
         ref char uplo,
         ref int n,
         ref complex_float AP, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpcon_")]
        public static extern void ZHPCON(
         ref char uplo,
         ref int n,
         ref complex_double AP, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpev_")]
        public static extern void CHPEV(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpev_")]
        public static extern void ZHPEV(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_double AP,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpevd_")]
        public static extern void CHPEVD(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpevd_")]
        public static extern void ZHPEVD(
         ref char jobz, ref char uplo,
         ref int n,
         ref complex_double AP,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpevx_")]
        public static extern void CHPEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work,
         ref float rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpevx_")]
        public static extern void ZHPEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_double AP,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work,
         ref double rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpgst_")]
        public static extern void CHPGST(
         ref int itype, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref complex_float BP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpgst_")]
        public static extern void ZHPGST(
         ref int itype, ref char uplo,
         ref int n,
         ref complex_double AP,
         ref complex_double BP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpgv_")]
        public static extern void CHPGV(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref complex_float BP,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpgv_")]
        public static extern void ZHPGV(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_double AP,
         ref complex_double BP,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpgvd_")]
        public static extern void CHPGVD(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref complex_float BP,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpgvd_")]
        public static extern void ZHPGVD(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref complex_double AP,
         ref complex_double BP,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpgvx_")]
        public static extern void CHPGVX(
         ref int itype, ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref complex_float BP,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work,
         ref float rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpgvx_")]
        public static extern void ZHPGVX(
         ref int itype, ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref complex_double AP,
         ref complex_double BP,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work,
         ref double rwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chprfs_")]
        public static extern void CHPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float AFP, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhprfs_")]
        public static extern void ZHPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double AFP, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpsv_")]
        public static extern void CHPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpsv_")]
        public static extern void ZHPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chpsvx_")]
        public static extern void CHPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float AFP, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhpsvx_")]
        public static extern void ZHPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double AFP, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chptrd_")]
        public static extern void CHPTRD(
         ref char uplo,
         ref int n,
         ref complex_float AP,
         ref float D,
         ref float E,
         ref complex_float tau,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhptrd_")]
        public static extern void ZHPTRD(
         ref char uplo,
         ref int n,
         ref complex_double AP,
         ref double D,
         ref double E,
         ref complex_double tau,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chptrf_")]
        public static extern void CHPTRF(
         ref char uplo,
         ref int n,
         ref complex_float AP, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhptrf_")]
        public static extern void ZHPTRF(
         ref char uplo,
         ref int n,
         ref complex_double AP, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chptri_")]
        public static extern void CHPTRI(
         ref char uplo,
         ref int n,
         ref complex_float AP, ref int ipiv,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhptri_")]
        public static extern void ZHPTRI(
         ref char uplo,
         ref int n,
         ref complex_double AP, ref int ipiv,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chptrs_")]
        public static extern void CHPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhptrs_")]
        public static extern void ZHPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chsein_")]
        public static extern void CHSEIN(
         ref char side, ref char eigsrc, ref char initv,
         ref int select,
         ref int n,
         ref complex_float H, ref int ldh,
         ref complex_float W,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr, ref int mm, ref int m,
         ref complex_float work,
         ref float rwork, ref int IFAILL, ref int IFAILR,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dhsein_")]
        public static extern void DHSEIN(
         ref char side, ref char eigsrc, ref char initv,
         ref int select,
         ref int n,
         ref double H, ref int ldh,
         ref double WR,
         ref double WI,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr, ref int mm, ref int m,
         ref double work, ref int IFAILL, ref int IFAILR,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "shsein_")]
        public static extern void SHSEIN(
         ref char side, ref char eigsrc, ref char initv,
         ref int select,
         ref int n,
         ref float H, ref int ldh,
         ref float WR,
         ref float WI,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr, ref int mm, ref int m,
         ref float work, ref int IFAILL, ref int IFAILR,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhsein_")]
        public static extern void ZHSEIN(
         ref char side, ref char eigsrc, ref char initv,
         ref int select,
         ref int n,
         ref complex_double H, ref int ldh,
         ref complex_double W,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr, ref int mm, ref int m,
         ref complex_double work,
         ref double rwork, ref int IFAILL, ref int IFAILR,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "chseqr_")]
        public static extern void CHSEQR(
         ref char job, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref complex_float H, ref int ldh,
         ref complex_float W,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dhseqr_")]
        public static extern void DHSEQR(
         ref char job, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref double H, ref int ldh,
         ref double WR,
         ref double WI,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "shseqr_")]
        public static extern void SHSEQR(
         ref char job, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref float H, ref int ldh,
         ref float WR,
         ref float WI,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zhseqr_")]
        public static extern void ZHSEQR(
         ref char job, ref char compz,
         ref int n, ref int ilo, ref int ihi,
         ref complex_double H, ref int ldh,
         ref complex_double W,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clacgv_")]
        public static extern void CLACGV(
         ref int n,
         ref complex_float X, ref int incx);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlacgv_")]
        public static extern void ZLACGV(
         ref int n,
         ref complex_double X, ref int incx);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clacn2_")]
        public static extern void CLACN2(
         ref int n,
         ref complex_float V,
         ref complex_float X,
         ref float est, ref int kase, ref int ISAVE);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlacn2_")]
        public static extern void DLACN2(
         ref int n,
         ref double V,
         ref double X, ref int ISGN,
         ref double est, ref int kase, ref int ISAVE);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slacn2_")]
        public static extern void SLACN2(
         ref int n,
         ref float V,
         ref float X, ref int ISGN,
         ref float est, ref int kase, ref int ISAVE);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlacn2_")]
        public static extern void ZLACN2(
         ref int n,
         ref complex_double V,
         ref complex_double X,
         ref double est, ref int kase, ref int ISAVE);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clacp2_")]
        public static extern void CLACP2(
         ref char uplo,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref complex_float B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlacp2_")]
        public static extern void ZLACP2(
         ref char uplo,
         ref int m, ref int n,
         ref double A, ref int lda,
         ref complex_double B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clacpy_")]
        public static extern void CLACPY(
         ref char uplo,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlacpy_")]
        public static extern void DLACPY(
         ref char uplo,
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slacpy_")]
        public static extern void SLACPY(
         ref char uplo,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlacpy_")]
        public static extern void ZLACPY(
         ref char uplo,
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clacrm_")]
        public static extern void CLACRM(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float B, ref int ldb,
         ref complex_float C, ref int ldc,
         ref float rwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlacrm_")]
        public static extern void ZLACRM(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref double B, ref int ldb,
         ref complex_double C, ref int ldc,
         ref double rwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlag2c_")]
        public static extern void ZLAG2C(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_float SA, ref int ldsa,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slag2d_")]
        public static extern void SLAG2D(
         ref int m, ref int n,
         ref float SA, ref int ldsa,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlag2s_")]
        public static extern void DLAG2S(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref float SA, ref int ldsa,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clag2z_")]
        public static extern void CLAG2Z(
         ref int m, ref int n,
         ref complex_float SA, ref int ldsa,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clagge_")]
        public static extern void CLAGGE(
         ref int m, ref int n, ref int kl, ref int ku,
         ref float D,
         ref complex_float A, ref int lda, ref int iseed,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlagge_")]
        public static extern void DLAGGE(
         ref int m, ref int n, ref int kl, ref int ku,
         ref double D,
         ref double A, ref int lda, ref int iseed,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slagge_")]
        public static extern void SLAGGE(
         ref int m, ref int n, ref int kl, ref int ku,
         ref float D,
         ref float A, ref int lda, ref int iseed,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlagge_")]
        public static extern void ZLAGGE(
         ref int m, ref int n, ref int kl, ref int ku,
         ref double D,
         ref complex_double A, ref int lda, ref int iseed,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "claghe_")]
        public static extern void CLAGHE(
         ref int n, ref int k,
         ref float D,
         ref complex_float A, ref int lda, ref int iseed,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlaghe_")]
        public static extern void ZLAGHE(
         ref int n, ref int k,
         ref double D,
         ref complex_double A, ref int lda, ref int iseed,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clagsy_")]
        public static extern void CLAGSY(
         ref int n, ref int k,
         ref float D,
         ref complex_float A, ref int lda, ref int iseed,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlagsy_")]
        public static extern void DLAGSY(
         ref int n, ref int k,
         ref double D,
         ref double A, ref int lda, ref int iseed,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slagsy_")]
        public static extern void SLAGSY(
         ref int n, ref int k,
         ref float D,
         ref float A, ref int lda, ref int iseed,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlagsy_")]
        public static extern void ZLAGSY(
         ref int n, ref int k,
         ref double D,
         ref complex_double A, ref int lda, ref int iseed,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlamch_")]
        public static extern double dlamch(
             ref char cmach);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slamch_")]
        public static extern float SLAMCH(
         ref char cmach);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clangb_")]
        public static extern float CLANGB(
         ref char norm,
         ref int n, ref int kl, ref int ku,
         ref complex_float AB, ref int ldab,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlangb(
             ref char norm,
             ref int n, ref int kl, ref int ku,
             ref double AB, ref int ldab,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slangb_")]
        public static extern float SLANGB(
         ref char norm,
         ref int n, ref int kl, ref int ku,
         ref float AB, ref int ldab,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlangb(
             ref char norm,
             ref int n, ref int kl, ref int ku,
             ref complex_double AB, ref int ldab,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clange_")]
        public static extern float CLANGE(
         ref char norm,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlange(
             ref char norm,
             ref int m, ref int n,
             ref double A, ref int lda,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slange_")]
        public static extern float SLANGE(
         ref char norm,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlange(
             ref char norm,
             ref int m, ref int n,
             ref complex_double A, ref int lda,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clangt_")]
        public static extern float CLANGT(
         ref char norm,
         ref int n,
         ref complex_float DL,
         ref complex_float D,
         ref complex_float DU);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlangt(
             ref char norm,
             ref int n,
             ref double DL,
             ref double D,
             ref double DU);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slangt_")]
        public static extern float SLANGT(
         ref char norm,
         ref int n,
         ref float DL,
         ref float D,
         ref float DU);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlangt(
             ref char norm,
             ref int n,
             ref complex_double DL,
             ref complex_double D,
             ref complex_double DU);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clanhb_")]
        public static extern float CLANHB(
         ref char norm, ref char uplo,
         ref int n, ref int k,
         ref complex_float AB, ref int ldab,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlanhb(
             ref char norm, ref char uplo,
             ref int n, ref int k,
             ref complex_double AB, ref int ldab,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clanhe_")]
        public static extern float CLANHE(
         ref char norm, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlanhe(
             ref char norm, ref char uplo,
             ref int n,
             ref complex_double A, ref int lda,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clanhp_")]
        public static extern float CLANHP(
         ref char norm, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlanhp(
             ref char norm, ref char uplo,
             ref int n,
             ref complex_double AP,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clanhs_")]
        public static extern float CLANHS(
         ref char norm,
         ref int n,
         ref complex_float A, ref int lda,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlanhs(
             ref char norm,
             ref int n,
             ref double A, ref int lda,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slanhs_")]
        public static extern float SLANHS(
         ref char norm,
         ref int n,
         ref float A, ref int lda,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlanhs(
             ref char norm,
             ref int n,
             ref complex_double A, ref int lda,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clanht_")]
        public static extern float CLANHT(
         ref char norm,
         ref int n,
         ref float D,
         ref complex_float E);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlanht(
             ref char norm,
             ref int n,
             ref double D,
             ref complex_double E);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clansb_")]
        public static extern float CLANSB(
         ref char norm, ref char uplo,
         ref int n, ref int k,
         ref complex_float AB, ref int ldab,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlansb(
             ref char norm, ref char uplo,
             ref int n, ref int k,
             ref double AB, ref int ldab,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slansb_")]
        public static extern float SLANSB(
         ref char norm, ref char uplo,
         ref int n, ref int k,
         ref float AB, ref int ldab,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlansb(
             ref char norm, ref char uplo,
             ref int n, ref int k,
             ref complex_double AB, ref int ldab,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clansp_")]
        public static extern float CLANSP(
         ref char norm, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlansp(
             ref char norm, ref char uplo,
             ref int n,
             ref double AP,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slansp_")]
        public static extern float SLANSP(
         ref char norm, ref char uplo,
         ref int n,
         ref float AP,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlansp(
             ref char norm, ref char uplo,
             ref int n,
             ref complex_double AP,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlanst(
             ref char norm,
             ref int n,
             ref double D,
             ref double E);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slanst_")]
        public static extern float SLANST(
         ref char norm,
         ref int n,
         ref float D,
         ref float E);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clansy_")]
        public static extern float CLANSY(
         ref char norm, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlansy(
             ref char norm, ref char uplo,
             ref int n,
             ref double A, ref int lda,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slansy_")]
        public static extern float SLANSY(
         ref char norm, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlansy(
             ref char norm, ref char uplo,
             ref int n,
             ref complex_double A, ref int lda,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clantb_")]
        public static extern float CLANTB(
         ref char norm, ref char uplo, ref char diag,
         ref int n, ref int k,
         ref complex_float AB, ref int ldab,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlantb(
             ref char norm, ref char uplo, ref char diag,
             ref int n, ref int k,
             ref double AB, ref int ldab,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slantb_")]
        public static extern float SLANTB(
         ref char norm, ref char uplo, ref char diag,
         ref int n, ref int k,
         ref float AB, ref int ldab,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlantb(
             ref char norm, ref char uplo, ref char diag,
             ref int n, ref int k,
             ref complex_double AB, ref int ldab,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clantp_")]
        public static extern float CLANTP(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref complex_float AP,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlantp(
             ref char norm, ref char uplo, ref char diag,
             ref int n,
             ref double AP,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slantp_")]
        public static extern float SLANTP(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref float AP,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlantp(
             ref char norm, ref char uplo, ref char diag,
             ref int n,
             ref complex_double AP,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clantr_")]
        public static extern float CLANTR(
         ref char norm, ref char uplo, ref char diag,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlantr(
             ref char norm, ref char uplo, ref char diag,
             ref int m, ref int n,
             ref double A, ref int lda,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slantr_")]
        public static extern float SLANTR(
         ref char norm, ref char uplo, ref char diag,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double zlantr(
             ref char norm, ref char uplo, ref char diag,
             ref int m, ref int n,
             ref complex_double A, ref int lda,
             ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clapmr_")]
        public static extern void CLAPMR(
         ref int forwrd, ref int m, ref int n,
         ref complex_float X, ref int ldx, ref int K);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlapmr_")]
        public static extern void DLAPMR(
         ref int forwrd, ref int m, ref int n,
         ref double X, ref int ldx, ref int K);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slapmr_")]
        public static extern void SLAPMR(
         ref int forwrd, ref int m, ref int n,
         ref float X, ref int ldx, ref int K);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlapmr_")]
        public static extern void ZLAPMR(
         ref int forwrd, ref int m, ref int n,
         ref complex_double X, ref int ldx, ref int K);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clapmt_")]
        public static extern void CLAPMT(
         ref int forwrd, ref int m, ref int n,
         ref complex_float X, ref int ldx, ref int K);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlapmt_")]
        public static extern void DLAPMT(
         ref int forwrd, ref int m, ref int n,
         ref double X, ref int ldx, ref int K);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slapmt_")]
        public static extern void SLAPMT(
         ref int forwrd, ref int m, ref int n,
         ref float X, ref int ldx, ref int K);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlapmt_")]
        public static extern void ZLAPMT(
         ref int forwrd, ref int m, ref int n,
         ref complex_double X, ref int ldx, ref int K);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlapy2(
             ref double x,
             ref double y);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slapy2_")]
        public static extern float SLAPY2(
         ref float x,
         ref float y);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "double_")]
        public static extern double dlapy3(
             ref double x,
             ref double y,
             ref double z);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slapy3_")]
        public static extern float SLAPY3(
         ref float x,
         ref float y,
         ref float z);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clarcm_")]
        public static extern void CLARCM(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float C, ref int ldc,
         ref float rwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlarcm_")]
        public static extern void ZLARCM(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double C, ref int ldc,
         ref double rwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clarf_")]
        public static extern void CLARF(
         ref char side,
         ref int m, ref int n,
         ref complex_float V, ref int incv,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlarf_")]
        public static extern void DLARF(
         ref char side,
         ref int m, ref int n,
         ref double V, ref int incv,
         ref double tau,
         ref double C, ref int ldc,
         ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slarf_")]
        public static extern void SLARF(
         ref char side,
         ref int m, ref int n,
         ref float V, ref int incv,
         ref float tau,
         ref float C, ref int ldc,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlarf_")]
        public static extern void ZLARF(
         ref char side,
         ref int m, ref int n,
         ref complex_double V, ref int incv,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clarfb_")]
        public static extern void CLARFB(
         ref char side, ref char trans, ref char direct, ref char storev,
         ref int m, ref int n, ref int k,
         ref complex_float V, ref int ldv,
         ref complex_float T, ref int ldt,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int ldwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlarfb_")]
        public static extern void DLARFB(
         ref char side, ref char trans, ref char direct, ref char storev,
         ref int m, ref int n, ref int k,
         ref double V, ref int ldv,
         ref double T, ref int ldt,
         ref double C, ref int ldc,
         ref double work, ref int ldwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slarfb_")]
        public static extern void SLARFB(
         ref char side, ref char trans, ref char direct, ref char storev,
         ref int m, ref int n, ref int k,
         ref float V, ref int ldv,
         ref float T, ref int ldt,
         ref float C, ref int ldc,
         ref float work, ref int ldwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlarfb_")]
        public static extern void ZLARFB(
         ref char side, ref char trans, ref char direct, ref char storev,
         ref int m, ref int n, ref int k,
         ref complex_double V, ref int ldv,
         ref complex_double T, ref int ldt,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int ldwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clarfg_")]
        public static extern void CLARFG(
         ref int n,
         ref complex_float alpha,
         ref complex_float X, ref int incx,
         ref complex_float tau);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlarfg_")]
        public static extern void DLARFG(
         ref int n,
         ref double alpha,
         ref double X, ref int incx,
         ref double tau);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slarfg_")]
        public static extern void SLARFG(
         ref int n,
         ref float alpha,
         ref float X, ref int incx,
         ref float tau);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlarfg_")]
        public static extern void ZLARFG(
         ref int n,
         ref complex_double alpha,
         ref complex_double X, ref int incx,
         ref complex_double tau);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clarft_")]
        public static extern void CLARFT(
         ref char direct, ref char storev,
         ref int n, ref int k,
         ref complex_float V, ref int ldv,
         ref complex_float tau,
         ref complex_float T, ref int ldt);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlarft_")]
        public static extern void DLARFT(
         ref char direct, ref char storev,
         ref int n, ref int k,
         ref double V, ref int ldv,
         ref double tau,
         ref double T, ref int ldt);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slarft_")]
        public static extern void SLARFT(
         ref char direct, ref char storev,
         ref int n, ref int k,
         ref float V, ref int ldv,
         ref float tau,
         ref float T, ref int ldt);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlarft_")]
        public static extern void ZLARFT(
         ref char direct, ref char storev,
         ref int n, ref int k,
         ref complex_double V, ref int ldv,
         ref complex_double tau,
         ref complex_double T, ref int ldt);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clarfx_")]
        public static extern void CLARFX(
         ref char side,
         ref int m, ref int n,
         ref complex_float V,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlarfx_")]
        public static extern void DLARFX(
         ref char side,
         ref int m, ref int n,
         ref double V,
         ref double tau,
         ref double C, ref int ldc,
         ref double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slarfx_")]
        public static extern void SLARFX(
         ref char side,
         ref int m, ref int n,
         ref float V,
         ref float tau,
         ref float C, ref int ldc,
         ref float work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlarfx_")]
        public static extern void ZLARFX(
         ref char side,
         ref int m, ref int n,
         ref complex_double V,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clarnv_")]
        public static extern void CLARNV(
         ref int idist, ref int iseed, ref int n,
         ref complex_float X);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlarnv_")]
        public static extern void DLARNV(
         ref int idist, ref int iseed, ref int n,
         ref double X);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slarnv_")]
        public static extern void SLARNV(
         ref int idist, ref int iseed, ref int n,
         ref float X);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlarnv_")]
        public static extern void ZLARNV(
         ref int idist, ref int iseed, ref int n,
         ref complex_double X);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlartgp_")]
        public static extern void DLARTGP(
         ref double f,
         ref double g,
         ref double cs,
         ref double sn,
         ref double r);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slartgp_")]
        public static extern void SLARTGP(
         ref float f,
         ref float g,
         ref float cs,
         ref float sn,
         ref float r);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlartgs_")]
        public static extern void DLARTGS(
         ref double x,
         ref double y,
         ref double sigma,
         ref double cs,
         ref double sn);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slartgs_")]
        public static extern void SLARTGS(
         ref float x,
         ref float y,
         ref float sigma,
         ref float cs,
         ref float sn);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clascl_")]
        public static extern void CLASCL(
         ref char type,
         ref int kl, ref int ku,
         ref float cfrom,
         ref float cto, ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlascl_")]
        public static extern void DLASCL(
         ref char type,
         ref int kl, ref int ku,
         ref double cfrom,
         ref double cto, ref int m, ref int n,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slascl_")]
        public static extern void SLASCL(
         ref char type,
         ref int kl, ref int ku,
         ref float cfrom,
         ref float cto, ref int m, ref int n,
         ref float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlascl_")]
        public static extern void ZLASCL(
         ref char type,
         ref int kl, ref int ku,
         ref double cfrom,
         ref double cto, ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "claset_")]
        public static extern void CLASET(
         ref char uplo,
         ref int m, ref int n,
         ref complex_float alpha,
         ref complex_float beta,
         ref complex_float A, ref int lda);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlaset_")]
        public static extern void DLASET(
         ref char uplo,
         ref int m, ref int n,
         ref double alpha,
         ref double beta,
         ref double A, ref int lda);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slaset_")]
        public static extern void SLASET(
         ref char uplo,
         ref int m, ref int n,
         ref float alpha,
         ref float beta,
         ref float A, ref int lda);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlaset_")]
        public static extern void ZLASET(
         ref char uplo,
         ref int m, ref int n,
         ref complex_double alpha,
         ref complex_double beta,
         ref complex_double A, ref int lda);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlasrt_")]
        public static extern void DLASRT(
         ref char id,
         ref int n,
         ref double D,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slasrt_")]
        public static extern void SLASRT(
         ref char id,
         ref int n,
         ref float D,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "classq_")]
        public static extern void CLASSQ(
         ref int n,
         ref complex_float X, ref int incx,
         ref float scale,
         ref float sumsq);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlassq_")]
        public static extern void DLASSQ(
         ref int n,
         ref double X, ref int incx,
         ref double scale,
         ref double sumsq);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slassq_")]
        public static extern void SLASSQ(
         ref int n,
         ref float X, ref int incx,
         ref float scale,
         ref float sumsq);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlassq_")]
        public static extern void ZLASSQ(
         ref int n,
         ref complex_double X, ref int incx,
         ref double scale,
         ref double sumsq);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "claswp_")]
        public static extern void CLASWP(
         ref int n,
         ref complex_float A, ref int lda, ref int k1, ref int k2, ref int ipiv, ref int incx);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlaswp_")]
        public static extern void DLASWP(
         ref int n,
         ref double A, ref int lda, ref int k1, ref int k2, ref int ipiv, ref int incx);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slaswp_")]
        public static extern void SLASWP(
         ref int n,
         ref float A, ref int lda, ref int k1, ref int k2, ref int ipiv, ref int incx);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlaswp_")]
        public static extern void ZLASWP(
         ref int n,
         ref complex_double A, ref int lda, ref int k1, ref int k2, ref int ipiv, ref int incx);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clatms_")]
        public static extern void CLATMS(
         ref int m, ref int n, ref char dist,
         ref int iseed, ref char sym,
         ref float D,
         ref int mode,
         ref float cond,
         ref float dmax, ref int kl, ref int ku, ref char pack,
         ref complex_float A,
         ref int lda,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlatms_")]
        public static extern void DLATMS(
         ref int m, ref int n, ref char dist,
         ref int iseed, ref char sym,
         ref double D,
         ref int mode,
         ref double cond,
         ref double dmax, ref int kl, ref int ku, ref char pack,
         ref double A,
         ref int lda,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slatms_")]
        public static extern void SLATMS(
         ref int m, ref int n, ref char dist,
         ref int iseed, ref char sym,
         ref float D,
         ref int mode,
         ref float cond,
         ref float dmax, ref int kl, ref int ku, ref char pack,
         ref float A,
         ref int lda,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlatms_")]
        public static extern void ZLATMS(
         ref int m, ref int n, ref char dist,
         ref int iseed, ref char sym,
         ref double D,
         ref int mode,
         ref double cond,
         ref double dmax, ref int kl, ref int ku, ref char pack,
         ref complex_double A,
         ref int lda,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "clauum_")]
        public static extern void CLAUUM(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dlauum_")]
        public static extern void DLAUUM(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "slauum_")]
        public static extern void SLAUUM(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zlauum_")]
        public static extern void ZLAUUM(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ilaver_")]
        public static extern void ILAVER(
         ref int vers_major, ref int vers_minor, ref int vers_patch);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dopgtr_")]
        public static extern void DOPGTR(
         ref char uplo,
         ref int n,
         ref double AP,
         ref double tau,
         ref double Q, ref int ldq,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sopgtr_")]
        public static extern void SOPGTR(
         ref char uplo,
         ref int n,
         ref float AP,
         ref float tau,
         ref float Q, ref int ldq,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dopmtr_")]
        public static extern void DOPMTR(
         ref char side, ref char uplo, ref char trans,
         ref int m, ref int n,
         ref double AP,
         ref double tau,
         ref double C, ref int ldc,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sopmtr_")]
        public static extern void SOPMTR(
         ref char side, ref char uplo, ref char trans,
         ref int m, ref int n,
         ref float AP,
         ref float tau,
         ref float C, ref int ldc,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorbdb_")]
        public static extern void DORBDB(
         ref char trans, ref char signs,
         ref int m, ref int p, ref int q,
         ref double X11, ref int ldx11,
         ref double X12, ref int ldx12,
         ref double X21, ref int ldx21,
         ref double X22, ref int ldx22,
         ref double theta,
         ref double phi,
         ref double TAUP1,
         ref double TAUP2,
         ref double TAUQ1,
         ref double TAUQ2,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorbdb_")]
        public static extern void SORBDB(
         ref char trans, ref char signs,
         ref int m, ref int p, ref int q,
         ref float X11, ref int ldx11,
         ref float X12, ref int ldx12,
         ref float X21, ref int ldx21,
         ref float X22, ref int ldx22,
         ref float theta,
         ref float phi,
         ref float TAUP1,
         ref float TAUP2,
         ref float TAUQ1,
         ref float TAUQ2,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorcsd_")]
        public static extern void DORCSD(
         ref char jobu1, ref char jobu2, ref char jobv1t, ref char jobv2t, ref char trans, ref char signs,
         ref int m, ref int p, ref int q,
         ref double X11, ref int ldx11,
         ref double X12, ref int ldx12,
         ref double X21, ref int ldx21,
         ref double X22, ref int ldx22,
         ref double theta,
         ref double U1, ref int ldu1,
         ref double U2, ref int ldu2,
         ref double V1T, ref int ldv1t,
         ref double V2T, ref int ldv2t,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorcsd_")]
        public static extern void SORCSD(
         ref char jobu1, ref char jobu2, ref char jobv1t, ref char jobv2t, ref char trans, ref char signs,
         ref int m, ref int p, ref int q,
         ref float X11, ref int ldx11,
         ref float X12, ref int ldx12,
         ref float X21, ref int ldx21,
         ref float X22, ref int ldx22,
         ref float theta,
         ref float U1, ref int ldu1,
         ref float U2, ref int ldu2,
         ref float V1T, ref int ldv1t,
         ref float V2T, ref int ldv2t,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorcsd2by1_")]
        public static extern void DORCSD2BY1(
         ref char jobu1, ref char jobu2, ref char jobv1t,
         ref int m, ref int p, ref int q,
         ref double X11, ref int ldx11,
         ref double X21, ref int ldx21,
         ref double theta,
         ref double U1, ref int ldu1,
         ref double U2, ref int ldu2,
         ref double V1T, ref int ldv1t,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorcsd2by1_")]
        public static extern void SORCSD2BY1(
         ref char jobu1, ref char jobu2, ref char jobv1t,
         ref int m, ref int p, ref int q,
         ref float X11, ref int ldx11,
         ref float X21, ref int ldx21,
         ref float theta,
         ref float U1, ref int ldu1,
         ref float U2, ref int ldu2,
         ref float V1T, ref int ldv1t,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorgbr_")]
        public static extern void DORGBR(
         ref char vect,
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorgbr_")]
        public static extern void SORGBR(
         ref char vect,
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorghr_")]
        public static extern void DORGHR(
         ref int n, ref int ilo, ref int ihi,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorghr_")]
        public static extern void SORGHR(
         ref int n, ref int ilo, ref int ihi,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorglq_")]
        public static extern void DORGLQ(
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorglq_")]
        public static extern void SORGLQ(
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorgql_")]
        public static extern void DORGQL(
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorgql_")]
        public static extern void SORGQL(
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorgqr_")]
        public static extern void DORGQR(
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorgqr_")]
        public static extern void SORGQR(
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorgrq_")]
        public static extern void DORGRQ(
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorgrq_")]
        public static extern void SORGRQ(
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dorgtr_")]
        public static extern void DORGTR(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sorgtr_")]
        public static extern void SORGTR(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dormbr_")]
        public static extern void DORMBR(
         ref char vect, ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sormbr_")]
        public static extern void SORMBR(
         ref char vect, ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dormhr_")]
        public static extern void DORMHR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int ilo, ref int ihi,
         ref double A, ref int lda,
         ref double tau,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sormhr_")]
        public static extern void SORMHR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int ilo, ref int ihi,
         ref float A, ref int lda,
         ref float tau,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dormlq_")]
        public static extern void DORMLQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sormlq_")]
        public static extern void SORMLQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dormql_")]
        public static extern void DORMQL(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sormql_")]
        public static extern void SORMQL(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dormqr_")]
        public static extern void DORMQR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sormqr_")]
        public static extern void SORMQR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dormrq_")]
        public static extern void DORMRQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref double A, ref int lda,
         ref double tau,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sormrq_")]
        public static extern void SORMRQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref float A, ref int lda,
         ref float tau,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dormrz_")]
        public static extern void DORMRZ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l,
         ref double A, ref int lda,
         ref double tau,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sormrz_")]
        public static extern void SORMRZ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l,
         ref float A, ref int lda,
         ref float tau,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dormtr_")]
        public static extern void DORMTR(
         ref char side, ref char uplo, ref char trans,
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double C, ref int ldc,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sormtr_")]
        public static extern void SORMTR(
         ref char side, ref char uplo, ref char trans,
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float C, ref int ldc,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpbcon_")]
        public static extern void CPBCON(
         ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpbcon_")]
        public static extern void DPBCON(
         ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spbcon_")]
        public static extern void SPBCON(
         ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpbcon_")]
        public static extern void ZPBCON(
         ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpbequ_")]
        public static extern void CPBEQU(
         ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref float S,
         ref float scond,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpbequ_")]
        public static extern void DPBEQU(
         ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double S,
         ref double scond,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spbequ_")]
        public static extern void SPBEQU(
         ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float S,
         ref float scond,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpbequ_")]
        public static extern void ZPBEQU(
         ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref double S,
         ref double scond,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpbrfs_")]
        public static extern void CPBRFS(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float AFB, ref int ldafb,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpbrfs_")]
        public static extern void DPBRFS(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref double AB, ref int ldab,
         ref double AFB, ref int ldafb,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spbrfs_")]
        public static extern void SPBRFS(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref float AB, ref int ldab,
         ref float AFB, ref int ldafb,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpbrfs_")]
        public static extern void ZPBRFS(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double AFB, ref int ldafb,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpbstf_")]
        public static extern void CPBSTF(
         ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpbstf_")]
        public static extern void DPBSTF(
         ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spbstf_")]
        public static extern void SPBSTF(
         ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpbstf_")]
        public static extern void ZPBSTF(
         ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpbsv_")]
        public static extern void CPBSV(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpbsv_")]
        public static extern void DPBSV(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref double AB, ref int ldab,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spbsv_")]
        public static extern void SPBSV(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref float AB, ref int ldab,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpbsv_")]
        public static extern void ZPBSV(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpbsvx_")]
        public static extern void CPBSVX(
         ref char fact, ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float AFB, ref int ldafb, ref char equed,
         ref float S,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpbsvx_")]
        public static extern void DPBSVX(
         ref char fact, ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref double AB, ref int ldab,
         ref double AFB, ref int ldafb, ref char equed,
         ref double S,
         ref double B,
         ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spbsvx_")]
        public static extern void SPBSVX(
         ref char fact, ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref float AB, ref int ldab,
         ref float AFB, ref int ldafb, ref char equed,
         ref float S,
         ref float B,
         ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpbsvx_")]
        public static extern void ZPBSVX(
         ref char fact, ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double AFB, ref int ldafb, ref char equed,
         ref double S,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpbtrf_")]
        public static extern void CPBTRF(
         ref char uplo,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpbtrf_")]
        public static extern void DPBTRF(
         ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spbtrf_")]
        public static extern void SPBTRF(
         ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpbtrf_")]
        public static extern void ZPBTRF(
         ref char uplo,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpbtrs_")]
        public static extern void CPBTRS(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpbtrs_")]
        public static extern void DPBTRS(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref double AB, ref int ldab,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spbtrs_")]
        public static extern void SPBTRS(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref float AB, ref int ldab,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpbtrs_")]
        public static extern void ZPBTRS(
         ref char uplo,
         ref int n, ref int kd, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpftrf_")]
        public static extern void CPFTRF(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_float A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpftrf_")]
        public static extern void DPFTRF(
         ref char transr, ref char uplo,
         ref int n,
         ref double A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spftrf_")]
        public static extern void SPFTRF(
         ref char transr, ref char uplo,
         ref int n,
         ref float A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpftrf_")]
        public static extern void ZPFTRF(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_double A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpftri_")]
        public static extern void CPFTRI(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_float A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpftri_")]
        public static extern void DPFTRI(
         ref char transr, ref char uplo,
         ref int n,
         ref double A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spftri_")]
        public static extern void SPFTRI(
         ref char transr, ref char uplo,
         ref int n,
         ref float A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpftri_")]
        public static extern void ZPFTRI(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_double A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpftrs_")]
        public static extern void CPFTRS(
         ref char transr, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpftrs_")]
        public static extern void DPFTRS(
         ref char transr, ref char uplo,
         ref int n, ref int nrhs,
         ref double A,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spftrs_")]
        public static extern void SPFTRS(
         ref char transr, ref char uplo,
         ref int n, ref int nrhs,
         ref float A,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpftrs_")]
        public static extern void ZPFTRS(
         ref char transr, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpocon_")]
        public static extern void CPOCON(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpocon_")]
        public static extern void DPOCON(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spocon_")]
        public static extern void SPOCON(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpocon_")]
        public static extern void ZPOCON(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpoequ_")]
        public static extern void CPOEQU(
         ref int n,
         ref complex_float A, ref int lda,
         ref float S,
         ref float scond,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpoequ_")]
        public static extern void DPOEQU(
         ref int n,
         ref double A, ref int lda,
         ref double S,
         ref double scond,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spoequ_")]
        public static extern void SPOEQU(
         ref int n,
         ref float A, ref int lda,
         ref float S,
         ref float scond,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpoequ_")]
        public static extern void ZPOEQU(
         ref int n,
         ref complex_double A, ref int lda,
         ref double S,
         ref double scond,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpoequb_")]
        public static extern void CPOEQUB(
         ref int n,
         ref complex_float A, ref int lda,
         ref float S,
         ref float scond,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpoequb_")]
        public static extern void DPOEQUB(
         ref int n,
         ref double A, ref int lda,
         ref double S,
         ref double scond,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spoequb_")]
        public static extern void SPOEQUB(
         ref int n,
         ref float A, ref int lda,
         ref float S,
         ref float scond,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpoequb_")]
        public static extern void ZPOEQUB(
         ref int n,
         ref complex_double A, ref int lda,
         ref double S,
         ref double scond,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cporfs_")]
        public static extern void CPORFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dporfs_")]
        public static extern void DPORFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sporfs_")]
        public static extern void SPORFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zporfs_")]
        public static extern void ZPORFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cporfsx_")]
        public static extern void CPORFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf,
         ref float S,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dporfsx_")]
        public static extern void DPORFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf,
         ref double S,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sporfsx_")]
        public static extern void SPORFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf,
         ref float S,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zporfsx_")]
        public static extern void ZPORFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf,
         ref double S,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cposv_")]
        public static extern void CPOSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dposv_")]
        public static extern void DPOSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sposv_")]
        public static extern void SPOSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zposv_")]
        public static extern void ZPOSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsposv_")]
        public static extern void DSPOSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double work,
         ref float swork, ref int iter,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zcposv_")]
        public static extern void ZCPOSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref complex_double work,
         ref complex_float swork,
         ref double rwork, ref int iter,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cposvx_")]
        public static extern void CPOSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref char equed,
         ref float S,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dposvx_")]
        public static extern void DPOSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref char equed,
         ref double S,
         ref double B,
         ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sposvx_")]
        public static extern void SPOSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref char equed,
         ref float S,
         ref float B,
         ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zposvx_")]
        public static extern void ZPOSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref char equed,
         ref double S,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cposvxx_")]
        public static extern void CPOSVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref char equed,
         ref float S,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float rpvgrw,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dposvxx_")]
        public static extern void DPOSVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref char equed,
         ref double S,
         ref double B,
         ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double rpvgrw,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sposvxx_")]
        public static extern void SPOSVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref char equed,
         ref float S,
         ref float B,
         ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float rpvgrw,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zposvxx_")]
        public static extern void ZPOSVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref char equed,
         ref double S,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double rpvgrw,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpotf2_")]
        public static extern void CPOTF2(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpotf2_")]
        public static extern void DPOTF2(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spotf2_")]
        public static extern void SPOTF2(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpotf2_")]
        public static extern void ZPOTF2(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpotrf_")]
        public static extern void CPOTRF(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpotrf_")]
        public static extern void DPOTRF(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spotrf_")]
        public static extern void SPOTRF(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpotrf_")]
        public static extern void ZPOTRF(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpotrf2_")]
        public static extern void CPOTRF2(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpotrf2_")]
        public static extern void DPOTRF2(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spotrf2_")]
        public static extern void SPOTRF2(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpotrf2_")]
        public static extern void ZPOTRF2(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpotri_")]
        public static extern void CPOTRI(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpotri_")]
        public static extern void DPOTRI(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spotri_")]
        public static extern void SPOTRI(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpotri_")]
        public static extern void ZPOTRI(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpotrs_")]
        public static extern void CPOTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpotrs_")]
        public static extern void DPOTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spotrs_")]
        public static extern void SPOTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpotrs_")]
        public static extern void ZPOTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cppcon_")]
        public static extern void CPPCON(
         ref char uplo,
         ref int n,
         ref complex_float AP,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dppcon_")]
        public static extern void DPPCON(
         ref char uplo,
         ref int n,
         ref double AP,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sppcon_")]
        public static extern void SPPCON(
         ref char uplo,
         ref int n,
         ref float AP,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zppcon_")]
        public static extern void ZPPCON(
         ref char uplo,
         ref int n,
         ref complex_double AP,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cppequ_")]
        public static extern void CPPEQU(
         ref char uplo,
         ref int n,
         ref complex_float AP,
         ref float S,
         ref float scond,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dppequ_")]
        public static extern void DPPEQU(
         ref char uplo,
         ref int n,
         ref double AP,
         ref double S,
         ref double scond,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sppequ_")]
        public static extern void SPPEQU(
         ref char uplo,
         ref int n,
         ref float AP,
         ref float S,
         ref float scond,
         ref float amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zppequ_")]
        public static extern void ZPPEQU(
         ref char uplo,
         ref int n,
         ref complex_double AP,
         ref double S,
         ref double scond,
         ref double amax,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpprfs_")]
        public static extern void CPPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float AFP,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpprfs_")]
        public static extern void DPPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double AP,
         ref double AFP,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spprfs_")]
        public static extern void SPPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float AP,
         ref float AFP,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpprfs_")]
        public static extern void ZPPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double AFP,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cppsv_")]
        public static extern void CPPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dppsv_")]
        public static extern void DPPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double AP,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sppsv_")]
        public static extern void SPPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float AP,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zppsv_")]
        public static extern void ZPPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cppsvx_")]
        public static extern void CPPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float AFP, ref char equed,
         ref float S,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dppsvx_")]
        public static extern void DPPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref double AP,
         ref double AFP, ref char equed,
         ref double S,
         ref double B,
         ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sppsvx_")]
        public static extern void SPPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref float AP,
         ref float AFP, ref char equed,
         ref float S,
         ref float B,
         ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zppsvx_")]
        public static extern void ZPPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double AFP, ref char equed,
         ref double S,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpptrf_")]
        public static extern void CPPTRF(
         ref char uplo,
         ref int n,
         ref complex_float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpptrf_")]
        public static extern void DPPTRF(
         ref char uplo,
         ref int n,
         ref double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spptrf_")]
        public static extern void SPPTRF(
         ref char uplo,
         ref int n,
         ref float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpptrf_")]
        public static extern void ZPPTRF(
         ref char uplo,
         ref int n,
         ref complex_double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpptri_")]
        public static extern void CPPTRI(
         ref char uplo,
         ref int n,
         ref complex_float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpptri_")]
        public static extern void DPPTRI(
         ref char uplo,
         ref int n,
         ref double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spptri_")]
        public static extern void SPPTRI(
         ref char uplo,
         ref int n,
         ref float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpptri_")]
        public static extern void ZPPTRI(
         ref char uplo,
         ref int n,
         ref complex_double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpptrs_")]
        public static extern void CPPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpptrs_")]
        public static extern void DPPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double AP,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spptrs_")]
        public static extern void SPPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float AP,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpptrs_")]
        public static extern void ZPPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpstrf_")]
        public static extern void CPSTRF(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int piv, ref int rank,
         ref float tol,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpstrf_")]
        public static extern void DPSTRF(
         ref char uplo,
         ref int n,
         ref double A, ref int lda, ref int piv, ref int rank,
         ref double tol,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spstrf_")]
        public static extern void SPSTRF(
         ref char uplo,
         ref int n,
         ref float A, ref int lda, ref int piv, ref int rank,
         ref float tol,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpstrf_")]
        public static extern void ZPSTRF(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int piv, ref int rank,
         ref double tol,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cptcon_")]
        public static extern void CPTCON(
         ref int n,
         ref float D,
         ref complex_float E,
         ref float anorm,
         ref float rcond,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dptcon_")]
        public static extern void DPTCON(
         ref int n,
         ref double D,
         ref double E,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sptcon_")]
        public static extern void SPTCON(
         ref int n,
         ref float D,
         ref float E,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zptcon_")]
        public static extern void ZPTCON(
         ref int n,
         ref double D,
         ref complex_double E,
         ref double anorm,
         ref double rcond,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpteqr_")]
        public static extern void CPTEQR(
         ref char compz,
         ref int n,
         ref float D,
         ref float E,
         ref complex_float Z, ref int ldz,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpteqr_")]
        public static extern void DPTEQR(
         ref char compz,
         ref int n,
         ref double D,
         ref double E,
         ref double Z, ref int ldz,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spteqr_")]
        public static extern void SPTEQR(
         ref char compz,
         ref int n,
         ref float D,
         ref float E,
         ref float Z, ref int ldz,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpteqr_")]
        public static extern void ZPTEQR(
         ref char compz,
         ref int n,
         ref double D,
         ref double E,
         ref complex_double Z, ref int ldz,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cptrfs_")]
        public static extern void CPTRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float D,
         ref complex_float E,
         ref float DF,
         ref complex_float EF,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dptrfs_")]
        public static extern void DPTRFS(
         ref int n, ref int nrhs,
         ref double D,
         ref double E,
         ref double DF,
         ref double EF,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sptrfs_")]
        public static extern void SPTRFS(
         ref int n, ref int nrhs,
         ref float D,
         ref float E,
         ref float DF,
         ref float EF,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zptrfs_")]
        public static extern void ZPTRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double D,
         ref complex_double E,
         ref double DF,
         ref complex_double EF,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cptsv_")]
        public static extern void CPTSV(
         ref int n, ref int nrhs,
         ref float D,
         ref complex_float E,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dptsv_")]
        public static extern void DPTSV(
         ref int n, ref int nrhs,
         ref double D,
         ref double E,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sptsv_")]
        public static extern void SPTSV(
         ref int n, ref int nrhs,
         ref float D,
         ref float E,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zptsv_")]
        public static extern void ZPTSV(
         ref int n, ref int nrhs,
         ref double D,
         ref complex_double E,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cptsvx_")]
        public static extern void CPTSVX(
         ref char fact,
         ref int n, ref int nrhs,
         ref float D,
         ref complex_float E,
         ref float DF,
         ref complex_float EF,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dptsvx_")]
        public static extern void DPTSVX(
         ref char fact,
         ref int n, ref int nrhs,
         ref double D,
         ref double E,
         ref double DF,
         ref double EF,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sptsvx_")]
        public static extern void SPTSVX(
         ref char fact,
         ref int n, ref int nrhs,
         ref float D,
         ref float E,
         ref float DF,
         ref float EF,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zptsvx_")]
        public static extern void ZPTSVX(
         ref char fact,
         ref int n, ref int nrhs,
         ref double D,
         ref complex_double E,
         ref double DF,
         ref complex_double EF,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpttrf_")]
        public static extern void CPTTRF(
         ref int n,
         ref float D,
         ref complex_float E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpttrf_")]
        public static extern void DPTTRF(
         ref int n,
         ref double D,
         ref double E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spttrf_")]
        public static extern void SPTTRF(
         ref int n,
         ref float D,
         ref float E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpttrf_")]
        public static extern void ZPTTRF(
         ref int n,
         ref double D,
         ref complex_double E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cpttrs_")]
        public static extern void CPTTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float D,
         ref complex_float E,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dpttrs_")]
        public static extern void DPTTRS(
         ref int n, ref int nrhs,
         ref double D,
         ref double E,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "spttrs_")]
        public static extern void SPTTRS(
         ref int n, ref int nrhs,
         ref float D,
         ref float E,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zpttrs_")]
        public static extern void ZPTTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double D,
         ref complex_double E,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbev_")]
        public static extern void DSBEV(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double W,
         ref double Z, ref int ldz,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbev_")]
        public static extern void SSBEV(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float W,
         ref float Z, ref int ldz,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbev_2stage_")]
        public static extern void DSBEV_2STAGE(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbev_2stage_")]
        public static extern void SSBEV_2STAGE(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbevd_")]
        public static extern void DSBEVD(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbevd_")]
        public static extern void SSBEVD(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbevd_2stage_")]
        public static extern void DSBEVD_2STAGE(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbevd_2stage_")]
        public static extern void SSBEVD_2STAGE(
         ref char jobz, ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbevx_")]
        public static extern void DSBEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double Q, ref int ldq,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz,
         ref double work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbevx_")]
        public static extern void SSBEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float Q, ref int ldq,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz,
         ref float work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbevx_2stage_")]
        public static extern void DSBEVX_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double Q, ref int ldq,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbevx_2stage_")]
        public static extern void SSBEVX_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float Q, ref int ldq,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbgst_")]
        public static extern void DSBGST(
         ref char vect, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref double AB, ref int ldab,
         ref double BB, ref int ldbb,
         ref double X, ref int ldx,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbgst_")]
        public static extern void SSBGST(
         ref char vect, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref float AB, ref int ldab,
         ref float BB, ref int ldbb,
         ref float X, ref int ldx,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbgv_")]
        public static extern void DSBGV(
         ref char jobz, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref double AB, ref int ldab,
         ref double BB, ref int ldbb,
         ref double W,
         ref double Z, ref int ldz,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbgv_")]
        public static extern void SSBGV(
         ref char jobz, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref float AB, ref int ldab,
         ref float BB, ref int ldbb,
         ref float W,
         ref float Z, ref int ldz,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbgvd_")]
        public static extern void DSBGVD(
         ref char jobz, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref double AB, ref int ldab,
         ref double BB, ref int ldbb,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbgvd_")]
        public static extern void SSBGVD(
         ref char jobz, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref float AB, ref int ldab,
         ref float BB, ref int ldbb,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbgvx_")]
        public static extern void DSBGVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref double AB, ref int ldab,
         ref double BB, ref int ldbb,
         ref double Q, ref int ldq,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz,
         ref double work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbgvx_")]
        public static extern void SSBGVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n, ref int ka, ref int kb,
         ref float AB, ref int ldab,
         ref float BB, ref int ldbb,
         ref float Q, ref int ldq,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz,
         ref float work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsbtrd_")]
        public static extern void DSBTRD(
         ref char vect, ref char uplo,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double D,
         ref double E,
         ref double Q, ref int ldq,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssbtrd_")]
        public static extern void SSBTRD(
         ref char vect, ref char uplo,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float D,
         ref float E,
         ref float Q, ref int ldq,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsfrk_")]
        public static extern void DSFRK(
         ref char transr, ref char uplo, ref char trans,
         ref int n, ref int k,
         ref double alpha,
         ref double A, ref int lda,
         ref double beta,
         ref double C);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssfrk_")]
        public static extern void SSFRK(
         ref char transr, ref char uplo, ref char trans,
         ref int n, ref int k,
         ref float alpha,
         ref float A, ref int lda,
         ref float beta,
         ref float C);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cspcon_")]
        public static extern void CSPCON(
         ref char uplo,
         ref int n,
         ref complex_float AP, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspcon_")]
        public static extern void DSPCON(
         ref char uplo,
         ref int n,
         ref double AP, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspcon_")]
        public static extern void SSPCON(
         ref char uplo,
         ref int n,
         ref float AP, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zspcon_")]
        public static extern void ZSPCON(
         ref char uplo,
         ref int n,
         ref complex_double AP, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspev_")]
        public static extern void DSPEV(
         ref char jobz, ref char uplo,
         ref int n,
         ref double AP,
         ref double W,
         ref double Z, ref int ldz,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspev_")]
        public static extern void SSPEV(
         ref char jobz, ref char uplo,
         ref int n,
         ref float AP,
         ref float W,
         ref float Z, ref int ldz,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspevd_")]
        public static extern void DSPEVD(
         ref char jobz, ref char uplo,
         ref int n,
         ref double AP,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspevd_")]
        public static extern void SSPEVD(
         ref char jobz, ref char uplo,
         ref int n,
         ref float AP,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspevx_")]
        public static extern void DSPEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref double AP,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz,
         ref double work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspevx_")]
        public static extern void SSPEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref float AP,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz,
         ref float work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspgst_")]
        public static extern void DSPGST(
         ref int itype, ref char uplo,
         ref int n,
         ref double AP,
         ref double BP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspgst_")]
        public static extern void SSPGST(
         ref int itype, ref char uplo,
         ref int n,
         ref float AP,
         ref float BP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspgv_")]
        public static extern void DSPGV(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref double AP,
         ref double BP,
         ref double W,
         ref double Z, ref int ldz,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspgv_")]
        public static extern void SSPGV(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref float AP,
         ref float BP,
         ref float W,
         ref float Z, ref int ldz,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspgvd_")]
        public static extern void DSPGVD(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref double AP,
         ref double BP,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspgvd_")]
        public static extern void SSPGVD(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref float AP,
         ref float BP,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspgvx_")]
        public static extern void DSPGVX(
         ref int itype, ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref double AP,
         ref double BP,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz,
         ref double work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspgvx_")]
        public static extern void SSPGVX(
         ref int itype, ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref float AP,
         ref float BP,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz,
         ref float work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csprfs_")]
        public static extern void CSPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float AFP, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsprfs_")]
        public static extern void DSPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double AP,
         ref double AFP, ref int ipiv,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssprfs_")]
        public static extern void SSPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float AP,
         ref float AFP, ref int ipiv,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsprfs_")]
        public static extern void ZSPRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double AFP, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cspsv_")]
        public static extern void CSPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspsv_")]
        public static extern void DSPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double AP, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspsv_")]
        public static extern void SSPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float AP, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zspsv_")]
        public static extern void ZSPSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cspsvx_")]
        public static extern void CSPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float AFP, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dspsvx_")]
        public static extern void DSPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref double AP,
         ref double AFP, ref int ipiv,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sspsvx_")]
        public static extern void SSPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref float AP,
         ref float AFP, ref int ipiv,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zspsvx_")]
        public static extern void ZSPSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double AFP, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsptrd_")]
        public static extern void DSPTRD(
         ref char uplo,
         ref int n,
         ref double AP,
         ref double D,
         ref double E,
         ref double tau,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssptrd_")]
        public static extern void SSPTRD(
         ref char uplo,
         ref int n,
         ref float AP,
         ref float D,
         ref float E,
         ref float tau,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csptrf_")]
        public static extern void CSPTRF(
         ref char uplo,
         ref int n,
         ref complex_float AP, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsptrf_")]
        public static extern void DSPTRF(
         ref char uplo,
         ref int n,
         ref double AP, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssptrf_")]
        public static extern void SSPTRF(
         ref char uplo,
         ref int n,
         ref float AP, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsptrf_")]
        public static extern void ZSPTRF(
         ref char uplo,
         ref int n,
         ref complex_double AP, ref int ipiv,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csptri_")]
        public static extern void CSPTRI(
         ref char uplo,
         ref int n,
         ref complex_float AP, ref int ipiv,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsptri_")]
        public static extern void DSPTRI(
         ref char uplo,
         ref int n,
         ref double AP, ref int ipiv,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssptri_")]
        public static extern void SSPTRI(
         ref char uplo,
         ref int n,
         ref float AP, ref int ipiv,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsptri_")]
        public static extern void ZSPTRI(
         ref char uplo,
         ref int n,
         ref complex_double AP, ref int ipiv,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csptrs_")]
        public static extern void CSPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float AP, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsptrs_")]
        public static extern void DSPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double AP, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssptrs_")]
        public static extern void SSPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float AP, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsptrs_")]
        public static extern void ZSPTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double AP, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dstebz_")]
        public static extern void DSTEBZ(
         ref char range, ref char order,
         ref int n,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol,
         ref double D,
         ref double E, ref int m, ref int nsplit,
         ref double W, ref int IBLOCK, ref int ISPLIT,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sstebz_")]
        public static extern void SSTEBZ(
         ref char range, ref char order,
         ref int n,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol,
         ref float D,
         ref float E, ref int m, ref int nsplit,
         ref float W, ref int IBLOCK, ref int ISPLIT,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cstedc_")]
        public static extern void CSTEDC(
         ref char compz,
         ref int n,
         ref float D,
         ref float E,
         ref complex_float Z, ref int ldz,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dstedc_")]
        public static extern void DSTEDC(
         ref char compz,
         ref int n,
         ref double D,
         ref double E,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sstedc_")]
        public static extern void SSTEDC(
         ref char compz,
         ref int n,
         ref float D,
         ref float E,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zstedc_")]
        public static extern void ZSTEDC(
         ref char compz,
         ref int n,
         ref double D,
         ref double E,
         ref complex_double Z, ref int ldz,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cstegr_")]
        public static extern void CSTEGR(
         ref char jobz, ref char range,
         ref int n,
         ref float D,
         ref float E,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz, ref int ISUPPZ,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dstegr_")]
        public static extern void DSTEGR(
         ref char jobz, ref char range,
         ref int n,
         ref double D,
         ref double E,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz, ref int ISUPPZ,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sstegr_")]
        public static extern void SSTEGR(
         ref char jobz, ref char range,
         ref int n,
         ref float D,
         ref float E,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz, ref int ISUPPZ,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zstegr_")]
        public static extern void ZSTEGR(
         ref char jobz, ref char range,
         ref int n,
         ref double D,
         ref double E,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz, ref int ISUPPZ,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cstein_")]
        public static extern void CSTEIN(
         ref int n,
         ref float D,
         ref float E, ref int m,
         ref float W, ref int IBLOCK, ref int ISPLIT,
         ref complex_float Z, ref int ldz,
         ref float work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dstein_")]
        public static extern void DSTEIN(
         ref int n,
         ref double D,
         ref double E, ref int m,
         ref double W, ref int IBLOCK, ref int ISPLIT,
         ref double Z, ref int ldz,
         ref double work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sstein_")]
        public static extern void SSTEIN(
         ref int n,
         ref float D,
         ref float E, ref int m,
         ref float W, ref int IBLOCK, ref int ISPLIT,
         ref float Z, ref int ldz,
         ref float work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zstein_")]
        public static extern void ZSTEIN(
         ref int n,
         ref double D,
         ref double E, ref int m,
         ref double W, ref int IBLOCK, ref int ISPLIT,
         ref complex_double Z, ref int ldz,
         ref double work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cstemr_")]
        public static extern void CSTEMR(
         ref char jobz, ref char range,
         ref int n,
         ref float D,
         ref float E,
         ref float vl,
         ref float vu, ref int il, ref int iu, ref int m,
         ref float W,
         ref complex_float Z, ref int ldz, ref int nzc, ref int ISUPPZ, ref int tryrac,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dstemr_")]
        public static extern void DSTEMR(
         ref char jobz, ref char range,
         ref int n,
         ref double D,
         ref double E,
         ref double vl,
         ref double vu, ref int il, ref int iu, ref int m,
         ref double W,
         ref double Z, ref int ldz, ref int nzc, ref int ISUPPZ, ref int tryrac,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sstemr_")]
        public static extern void SSTEMR(
         ref char jobz, ref char range,
         ref int n,
         ref float D,
         ref float E,
         ref float vl,
         ref float vu, ref int il, ref int iu, ref int m,
         ref float W,
         ref float Z, ref int ldz, ref int nzc, ref int ISUPPZ, ref int tryrac,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zstemr_")]
        public static extern void ZSTEMR(
         ref char jobz, ref char range,
         ref int n,
         ref double D,
         ref double E,
         ref double vl,
         ref double vu, ref int il, ref int iu, ref int m,
         ref double W,
         ref complex_double Z, ref int ldz, ref int nzc, ref int ISUPPZ, ref int tryrac,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csteqr_")]
        public static extern void CSTEQR(
         ref char compz,
         ref int n,
         ref float D,
         ref float E,
         ref complex_float Z, ref int ldz,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsteqr_")]
        public static extern void DSTEQR(
         ref char compz,
         ref int n,
         ref double D,
         ref double E,
         ref double Z, ref int ldz,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssteqr_")]
        public static extern void SSTEQR(
         ref char compz,
         ref int n,
         ref float D,
         ref float E,
         ref float Z, ref int ldz,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsteqr_")]
        public static extern void ZSTEQR(
         ref char compz,
         ref int n,
         ref double D,
         ref double E,
         ref complex_double Z, ref int ldz,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsterf_")]
        public static extern void DSTERF(
         ref int n,
         ref double D,
         ref double E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssterf_")]
        public static extern void SSTERF(
         ref int n,
         ref float D,
         ref float E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dstev_")]
        public static extern void DSTEV(
         ref char jobz,
         ref int n,
         ref double D,
         ref double E,
         ref double Z, ref int ldz,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sstev_")]
        public static extern void SSTEV(
         ref char jobz,
         ref int n,
         ref float D,
         ref float E,
         ref float Z, ref int ldz,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dstevd_")]
        public static extern void DSTEVD(
         ref char jobz,
         ref int n,
         ref double D,
         ref double E,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sstevd_")]
        public static extern void SSTEVD(
         ref char jobz,
         ref int n,
         ref float D,
         ref float E,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dstevr_")]
        public static extern void DSTEVR(
         ref char jobz, ref char range,
         ref int n,
         ref double D,
         ref double E,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz, ref int ISUPPZ,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sstevr_")]
        public static extern void SSTEVR(
         ref char jobz, ref char range,
         ref int n,
         ref float D,
         ref float E,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz, ref int ISUPPZ,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dstevx_")]
        public static extern void DSTEVX(
         ref char jobz, ref char range,
         ref int n,
         ref double D,
         ref double E,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz,
         ref double work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "sstevx_")]
        public static extern void SSTEVX(
         ref char jobz, ref char range,
         ref int n,
         ref float D,
         ref float E,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz,
         ref float work,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csycon_")]
        public static extern void CSYCON(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsycon_")]
        public static extern void DSYCON(
         ref char uplo,
         ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssycon_")]
        public static extern void SSYCON(
         ref char uplo,
         ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsycon_")]
        public static extern void ZSYCON(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csycon_3_")]
        public static extern void CSYCON_3(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsycon_3_")]
        public static extern void DSYCON_3(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double E, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssycon_3_")]
        public static extern void SSYCON_3(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float E, ref int ipiv,
         ref float anorm,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsycon_3_")]
        public static extern void ZSYCON_3(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref double anorm,
         ref double rcond,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csyconv_")]
        public static extern void CSYCONV(
         ref char uplo, ref char way,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyconv_")]
        public static extern void DSYCONV(
         ref char uplo, ref char way,
         ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref double E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyconv_")]
        public static extern void SSYCONV(
         ref char uplo, ref char way,
         ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref float E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsyconv_")]
        public static extern void ZSYCONV(
         ref char uplo, ref char way,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double E,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csyequb_")]
        public static extern void CSYEQUB(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref float S,
         ref float scond,
         ref float amax,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyequb_")]
        public static extern void DSYEQUB(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double S,
         ref double scond,
         ref double amax,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyequb_")]
        public static extern void SSYEQUB(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float S,
         ref float scond,
         ref float amax,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsyequb_")]
        public static extern void ZSYEQUB(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref double S,
         ref double scond,
         ref double amax,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyev_")]
        public static extern void DSYEV(
         ref char jobz, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double W,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyev_")]
        public static extern void SSYEV(
         ref char jobz, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float W,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyev_2stage_")]
        public static extern void DSYEV_2STAGE(
         ref char jobz, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double W,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyev_2stage_")]
        public static extern void SSYEV_2STAGE(
         ref char jobz, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float W,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyevd_")]
        public static extern void DSYEVD(
         ref char jobz, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double W,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyevd_")]
        public static extern void SSYEVD(
         ref char jobz, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float W,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyevd_2stage_")]
        public static extern void DSYEVD_2STAGE(
         ref char jobz, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double W,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyevd_2stage_")]
        public static extern void SSYEVD_2STAGE(
         ref char jobz, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float W,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyevr_")]
        public static extern void DSYEVR(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz, ref int ISUPPZ,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyevr_")]
        public static extern void SSYEVR(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz, ref int ISUPPZ,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyevr_2stage_")]
        public static extern void DSYEVR_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz, ref int ISUPPZ,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyevr_2stage_")]
        public static extern void SSYEVR_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz, ref int ISUPPZ,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyevx_")]
        public static extern void DSYEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyevx_")]
        public static extern void SSYEVX(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyevx_2stage_")]
        public static extern void DSYEVX_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyevx_2stage_")]
        public static extern void SSYEVX_2STAGE(
         ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsygst_")]
        public static extern void DSYGST(
         ref int itype, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssygst_")]
        public static extern void SSYGST(
         ref int itype, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsygv_")]
        public static extern void DSYGV(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double W,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssygv_")]
        public static extern void SSYGV(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float W,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsygv_2stage_")]
        public static extern void DSYGV_2STAGE(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double W,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssygv_2stage_")]
        public static extern void SSYGV_2STAGE(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float W,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsygvd_")]
        public static extern void DSYGVD(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double W,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssygvd_")]
        public static extern void SSYGVD(
         ref int itype, ref char jobz, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float W,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsygvx_")]
        public static extern void DSYGVX(
         ref int itype, ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double vl,
         ref double vu, ref int il, ref int iu,
         ref double abstol, ref int m,
         ref double W,
         ref double Z, ref int ldz,
         ref double work, ref int lwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssygvx_")]
        public static extern void SSYGVX(
         ref int itype, ref char jobz, ref char range, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float vl,
         ref float vu, ref int il, ref int iu,
         ref float abstol, ref int m,
         ref float W,
         ref float Z, ref int ldz,
         ref float work, ref int lwork,
         ref int iwork, ref int IFAIL,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csyr_")]
        public static extern void CSYR(
         ref char uplo,
         ref int n,
         ref complex_float alpha,
         ref complex_float X, ref int incx,
         ref complex_float A, ref int lda);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsyr_")]
        public static extern void ZSYR(
         ref char uplo,
         ref int n,
         ref complex_double alpha,
         ref complex_double X, ref int incx,
         ref complex_double A, ref int lda);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csyrfs_")]
        public static extern void CSYRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyrfs_")]
        public static extern void DSYRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref int ipiv,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyrfs_")]
        public static extern void SSYRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref int ipiv,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsyrfs_")]
        public static extern void ZSYRFS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csyrfsx_")]
        public static extern void CSYRFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv,
         ref float S,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyrfsx_")]
        public static extern void DSYRFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref int ipiv,
         ref double S,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyrfsx_")]
        public static extern void SSYRFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref int ipiv,
         ref float S,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsyrfsx_")]
        public static extern void ZSYRFSX(
         ref char uplo, ref char equed,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv,
         ref double S,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csysv_")]
        public static extern void CSYSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsysv_")]
        public static extern void DSYSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssysv_")]
        public static extern void SSYSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda, ref int ipiv,
         ref float B, ref int ldb,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsysv_")]
        public static extern void ZSYSV(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csysv_aa_")]
        public static extern void CSYSV_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsysv_aa_")]
        public static extern void DSYSV_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssysv_aa_")]
        public static extern void SSYSV_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda, ref int ipiv,
         ref float B, ref int ldb,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsysv_aa_")]
        public static extern void ZSYSV_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csysv_aa_2stage_")]
        public static extern void CSYSV_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsysv_aa_2stage_")]
        public static extern void DSYSV_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref double B, ref int ldb,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssysv_aa_2stage_")]
        public static extern void SSYSV_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref float B, ref int ldb,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsysv_aa_2stage_")]
        public static extern void ZSYSV_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csysv_rk_")]
        public static extern void CSYSV_RK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsysv_rk_")]
        public static extern void DSYSV_RK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double E, ref int ipiv,
         ref double B, ref int ldb,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssysv_rk_")]
        public static extern void SSYSV_RK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float E, ref int ipiv,
         ref float B, ref int ldb,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsysv_rk_")]
        public static extern void ZSYSV_RK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csysv_rook_")]
        public static extern void CSYSV_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsysv_rook_")]
        public static extern void DSYSV_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssysv_rook_")]
        public static extern void SSYSV_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda, ref int ipiv,
         ref float B, ref int ldb,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsysv_rook_")]
        public static extern void ZSYSV_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csysvx_")]
        public static extern void CSYSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref complex_float work, ref int lwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsysvx_")]
        public static extern void DSYSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref int ipiv,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssysvx_")]
        public static extern void SSYSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref int ipiv,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float ferr,
         ref float berr,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsysvx_")]
        public static extern void ZSYSVX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double ferr,
         ref double berr,
         ref complex_double work, ref int lwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csysvxx_")]
        public static extern void CSYSVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float AF, ref int ldaf, ref int ipiv, ref char equed,
         ref float S,
         ref complex_float B,
         ref int ldb,
         ref complex_float X, ref int ldx,
         ref float rcond,
         ref float rpvgrw,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsysvxx_")]
        public static extern void DSYSVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double AF, ref int ldaf, ref int ipiv, ref char equed,
         ref double S,
         ref double B,
         ref int ldb,
         ref double X, ref int ldx,
         ref double rcond,
         ref double rpvgrw,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssysvxx_")]
        public static extern void SSYSVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float AF, ref int ldaf, ref int ipiv, ref char equed,
         ref float S,
         ref float B,
         ref int ldb,
         ref float X, ref int ldx,
         ref float rcond,
         ref float rpvgrw,
         ref float berr, ref int n_err_bnds,
         ref float err_bnds_norm,
         ref float err_bnds_comp, ref int nparams_,
         ref float params_,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsysvxx_")]
        public static extern void ZSYSVXX(
         ref char fact, ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double AF, ref int ldaf, ref int ipiv, ref char equed,
         ref double S,
         ref complex_double B,
         ref int ldb,
         ref complex_double X, ref int ldx,
         ref double rcond,
         ref double rpvgrw,
         ref double berr, ref int n_err_bnds,
         ref double err_bnds_norm,
         ref double err_bnds_comp, ref int nparams_,
         ref double params_,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csyswapr_")]
        public static extern void CSYSWAPR(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int i1, ref int i2);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsyswapr_")]
        public static extern void DSYSWAPR(
         ref char uplo,
         ref int n,
         ref double A, ref int lda, ref int i1, ref int i2);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssyswapr_")]
        public static extern void SSYSWAPR(
         ref char uplo,
         ref int n,
         ref float A, ref int lda, ref int i1, ref int i2);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsyswapr_")]
        public static extern void ZSYSWAPR(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int i1, ref int i2);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrd_")]
        public static extern void DSYTRD(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double D,
         ref double E,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrd_")]
        public static extern void SSYTRD(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float D,
         ref float E,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrd_2stage_")]
        public static extern void DSYTRD_2STAGE(
         ref char vect, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double D,
         ref double E,
         ref double tau,
         ref double HOUS2, ref int lhous2,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrd_2stage_")]
        public static extern void SSYTRD_2STAGE(
         ref char vect, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float D,
         ref float E,
         ref float tau,
         ref float HOUS2, ref int lhous2,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrf_")]
        public static extern void CSYTRF(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrf_")]
        public static extern void DSYTRF(
         ref char uplo,
         ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrf_")]
        public static extern void SSYTRF(
         ref char uplo,
         ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrf_")]
        public static extern void ZSYTRF(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrf_aa_")]
        public static extern void CSYTRF_AA(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrf_aa_")]
        public static extern void DSYTRF_AA(
         ref char uplo,
         ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrf_aa_")]
        public static extern void SSYTRF_AA(
         ref char uplo,
         ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrf_aa_")]
        public static extern void ZSYTRF_AA(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrf_aa_2stage_")]
        public static extern void CSYTRF_AA_2STAGE(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrf_aa_2stage_")]
        public static extern void DSYTRF_AA_2STAGE(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrf_aa_2stage_")]
        public static extern void SSYTRF_AA_2STAGE(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrf_aa_2stage_")]
        public static extern void ZSYTRF_AA_2STAGE(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrf_rk_")]
        public static extern void CSYTRF_RK(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrf_rk_")]
        public static extern void DSYTRF_RK(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double E, ref int ipiv,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrf_rk_")]
        public static extern void SSYTRF_RK(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float E, ref int ipiv,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrf_rk_")]
        public static extern void ZSYTRF_RK(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrf_rook_")]
        public static extern void CSYTRF_ROOK(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrf_rook_")]
        public static extern void DSYTRF_ROOK(
         ref char uplo,
         ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrf_rook_")]
        public static extern void SSYTRF_ROOK(
         ref char uplo,
         ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrf_rook_")]
        public static extern void ZSYTRF_ROOK(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytri_")]
        public static extern void CSYTRI(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytri_")]
        public static extern void DSYTRI(
         ref char uplo,
         ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytri_")]
        public static extern void SSYTRI(
         ref char uplo,
         ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytri_")]
        public static extern void ZSYTRI(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytri2_")]
        public static extern void CSYTRI2(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytri2_")]
        public static extern void DSYTRI2(
         ref char uplo,
         ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytri2_")]
        public static extern void SSYTRI2(
         ref char uplo,
         ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytri2_")]
        public static extern void ZSYTRI2(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytri2x_")]
        public static extern void CSYTRI2X(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float work, ref int nb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytri2x_")]
        public static extern void DSYTRI2X(
         ref char uplo,
         ref int n,
         ref double A, ref int lda, ref int ipiv,
         ref double work, ref int nb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytri2x_")]
        public static extern void SSYTRI2X(
         ref char uplo,
         ref int n,
         ref float A, ref int lda, ref int ipiv,
         ref float work, ref int nb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytri2x_")]
        public static extern void ZSYTRI2X(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double work, ref int nb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytri_3_")]
        public static extern void CSYTRI_3(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytri_3_")]
        public static extern void DSYTRI_3(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double E, ref int ipiv,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytri_3_")]
        public static extern void SSYTRI_3(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float E, ref int ipiv,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytri_3_")]
        public static extern void ZSYTRI_3(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrs_")]
        public static extern void CSYTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrs_")]
        public static extern void DSYTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrs_")]
        public static extern void SSYTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrs_")]
        public static extern void ZSYTRS(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrs2_")]
        public static extern void CSYTRS2(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrs2_")]
        public static extern void DSYTRS2(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrs2_")]
        public static extern void SSYTRS2(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda, ref int ipiv,
         ref float B, ref int ldb,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrs2_")]
        public static extern void ZSYTRS2(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrs_3_")]
        public static extern void CSYTRS_3(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float E, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrs_3_")]
        public static extern void DSYTRS_3(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double E, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrs_3_")]
        public static extern void SSYTRS_3(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float E, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrs_3_")]
        public static extern void ZSYTRS_3(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double E, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrs_aa_")]
        public static extern void CSYTRS_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrs_aa_")]
        public static extern void DSYTRS_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrs_aa_")]
        public static extern void SSYTRS_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda, ref int ipiv,
         ref float B, ref int ldb,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrs_aa_")]
        public static extern void ZSYTRS_AA(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrs_aa_2stage_")]
        public static extern void CSYTRS_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrs_aa_2stage_")]
        public static extern void DSYTRS_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrs_aa_2stage_")]
        public static extern void SSYTRS_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrs_aa_2stage_")]
        public static extern void ZSYTRS_AA_2STAGE(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double TB, ref int ltb, ref int ipiv, ref int ipiv2,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "csytrs_rook_")]
        public static extern void CSYTRS_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda, ref int ipiv,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dsytrs_rook_")]
        public static extern void DSYTRS_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref double A, ref int lda, ref int ipiv,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ssytrs_rook_")]
        public static extern void SSYTRS_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref float A, ref int lda, ref int ipiv,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zsytrs_rook_")]
        public static extern void ZSYTRS_ROOK(
         ref char uplo,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda, ref int ipiv,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctbcon_")]
        public static extern void CTBCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n, ref int kd,
         ref complex_float AB, ref int ldab,
         ref float rcond,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtbcon_")]
        public static extern void DTBCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n, ref int kd,
         ref double AB, ref int ldab,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stbcon_")]
        public static extern void STBCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n, ref int kd,
         ref float AB, ref int ldab,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztbcon_")]
        public static extern void ZTBCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n, ref int kd,
         ref complex_double AB, ref int ldab,
         ref double rcond,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctbrfs_")]
        public static extern void CTBRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int kd, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtbrfs_")]
        public static extern void DTBRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int kd, ref int nrhs,
         ref double AB, ref int ldab,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stbrfs_")]
        public static extern void STBRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int kd, ref int nrhs,
         ref float AB, ref int ldab,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztbrfs_")]
        public static extern void ZTBRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int kd, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctbtrs_")]
        public static extern void CTBTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int kd, ref int nrhs,
         ref complex_float AB, ref int ldab,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtbtrs_")]
        public static extern void DTBTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int kd, ref int nrhs,
         ref double AB, ref int ldab,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stbtrs_")]
        public static extern void STBTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int kd, ref int nrhs,
         ref float AB, ref int ldab,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztbtrs_")]
        public static extern void ZTBTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int kd, ref int nrhs,
         ref complex_double AB, ref int ldab,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctfsm_")]
        public static extern void CTFSM(
         ref char transr, ref char side, ref char uplo, ref char trans, ref char diag,
         ref int m, ref int n,
         ref complex_float alpha,
         ref complex_float A,
         ref complex_float B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtfsm_")]
        public static extern void DTFSM(
         ref char transr, ref char side, ref char uplo, ref char trans, ref char diag,
         ref int m, ref int n,
         ref double alpha,
         ref double A,
         ref double B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stfsm_")]
        public static extern void STFSM(
         ref char transr, ref char side, ref char uplo, ref char trans, ref char diag,
         ref int m, ref int n,
         ref float alpha,
         ref float A,
         ref float B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztfsm_")]
        public static extern void ZTFSM(
         ref char transr, ref char side, ref char uplo, ref char trans, ref char diag,
         ref int m, ref int n,
         ref complex_double alpha,
         ref complex_double A,
         ref complex_double B, ref int ldb);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctftri_")]
        public static extern void CTFTRI(
         ref char transr, ref char uplo, ref char diag,
         ref int n,
         ref complex_float A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtftri_")]
        public static extern void DTFTRI(
         ref char transr, ref char uplo, ref char diag,
         ref int n,
         ref double A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stftri_")]
        public static extern void STFTRI(
         ref char transr, ref char uplo, ref char diag,
         ref int n,
         ref float A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztftri_")]
        public static extern void ZTFTRI(
         ref char transr, ref char uplo, ref char diag,
         ref int n,
         ref complex_double A,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctfttp_")]
        public static extern void CTFTTP(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_float ARF,
         ref complex_float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtfttp_")]
        public static extern void DTFTTP(
         ref char transr, ref char uplo,
         ref int n,
         ref double ARF,
         ref double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stfttp_")]
        public static extern void STFTTP(
         ref char transr, ref char uplo,
         ref int n,
         ref float ARF,
         ref float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztfttp_")]
        public static extern void ZTFTTP(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_double ARF,
         ref complex_double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctfttr_")]
        public static extern void CTFTTR(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_float ARF,
         ref complex_float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtfttr_")]
        public static extern void DTFTTR(
         ref char transr, ref char uplo,
         ref int n,
         ref double ARF,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stfttr_")]
        public static extern void STFTTR(
         ref char transr, ref char uplo,
         ref int n,
         ref float ARF,
         ref float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztfttr_")]
        public static extern void ZTFTTR(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_double ARF,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctgevc_")]
        public static extern void CTGEVC(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref complex_float S, ref int lds,
         ref complex_float P, ref int ldp,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr, ref int mm, ref int m,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtgevc_")]
        public static extern void DTGEVC(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref double S, ref int lds,
         ref double P, ref int ldp,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr, ref int mm, ref int m,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stgevc_")]
        public static extern void STGEVC(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref float S, ref int lds,
         ref float P, ref int ldp,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr, ref int mm, ref int m,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztgevc_")]
        public static extern void ZTGEVC(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref complex_double S, ref int lds,
         ref complex_double P, ref int ldp,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr, ref int mm, ref int m,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctgexc_")]
        public static extern void CTGEXC(
         ref int wantq, ref int wantz, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float Q, ref int ldq,
         ref complex_float Z, ref int ldz, ref int ifst, ref int ilst,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtgexc_")]
        public static extern void DTGEXC(
         ref int wantq, ref int wantz, ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double Q, ref int ldq,
         ref double Z, ref int ldz, ref int ifst, ref int ilst,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stgexc_")]
        public static extern void STGEXC(
         ref int wantq, ref int wantz, ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float Q, ref int ldq,
         ref float Z, ref int ldz, ref int ifst, ref int ilst,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztgexc_")]
        public static extern void ZTGEXC(
         ref int wantq, ref int wantz, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double Q, ref int ldq,
         ref complex_double Z, ref int ldz, ref int ifst, ref int ilst,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctgsen_")]
        public static extern void CTGSEN(
         ref int ijob, ref int wantq, ref int wantz, ref int select, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float alpha,
         ref complex_float beta,
         ref complex_float Q, ref int ldq,
         ref complex_float Z, ref int ldz, ref int m,
         ref float pl,
         ref float pr,
         ref float DIF,
         ref complex_float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtgsen_")]
        public static extern void DTGSEN(
         ref int ijob, ref int wantq, ref int wantz, ref int select, ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double alphar,
         ref double alphai,
         ref double beta,
         ref double Q, ref int ldq,
         ref double Z, ref int ldz, ref int m,
         ref double pl,
         ref double pr,
         ref double DIF,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stgsen_")]
        public static extern void STGSEN(
         ref int ijob, ref int wantq, ref int wantz, ref int select, ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float alphar,
         ref float alphai,
         ref float beta,
         ref float Q, ref int ldq,
         ref float Z, ref int ldz, ref int m,
         ref float pl,
         ref float pr,
         ref float DIF,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztgsen_")]
        public static extern void ZTGSEN(
         ref int ijob, ref int wantq, ref int wantz, ref int select, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double alpha,
         ref complex_double beta,
         ref complex_double Q, ref int ldq,
         ref complex_double Z, ref int ldz, ref int m,
         ref double pl,
         ref double pr,
         ref double DIF,
         ref complex_double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctgsja_")]
        public static extern void CTGSJA(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n, ref int k, ref int l,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref float tola,
         ref float tolb,
         ref float alpha,
         ref float beta,
         ref complex_float U, ref int ldu,
         ref complex_float V, ref int ldv,
         ref complex_float Q, ref int ldq,
         ref complex_float work, ref int ncycle,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtgsja_")]
        public static extern void DTGSJA(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n, ref int k, ref int l,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double tola,
         ref double tolb,
         ref double alpha,
         ref double beta,
         ref double U, ref int ldu,
         ref double V, ref int ldv,
         ref double Q, ref int ldq,
         ref double work, ref int ncycle,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stgsja_")]
        public static extern void STGSJA(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n, ref int k, ref int l,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float tola,
         ref float tolb,
         ref float alpha,
         ref float beta,
         ref float U, ref int ldu,
         ref float V, ref int ldv,
         ref float Q, ref int ldq,
         ref float work, ref int ncycle,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztgsja_")]
        public static extern void ZTGSJA(
         ref char jobu, ref char jobv, ref char jobq,
         ref int m, ref int p, ref int n, ref int k, ref int l,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref double tola,
         ref double tolb,
         ref double alpha,
         ref double beta,
         ref complex_double U, ref int ldu,
         ref complex_double V, ref int ldv,
         ref complex_double Q, ref int ldq,
         ref complex_double work, ref int ncycle,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctgsna_")]
        public static extern void CTGSNA(
         ref char job, ref char howmny,
         ref int select,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr,
         ref float S,
         ref float DIF, ref int mm, ref int m,
         ref complex_float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtgsna_")]
        public static extern void DTGSNA(
         ref char job, ref char howmny,
         ref int select,
         ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr,
         ref double S,
         ref double DIF, ref int mm, ref int m,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stgsna_")]
        public static extern void STGSNA(
         ref char job, ref char howmny,
         ref int select,
         ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr,
         ref float S,
         ref float DIF, ref int mm, ref int m,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztgsna_")]
        public static extern void ZTGSNA(
         ref char job, ref char howmny,
         ref int select,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr,
         ref double S,
         ref double DIF, ref int mm, ref int m,
         ref complex_double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctgsyl_")]
        public static extern void CTGSYL(
         ref char trans,
         ref int ijob, ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float C, ref int ldc,
         ref complex_float D, ref int ldd,
         ref complex_float E, ref int lde,
         ref complex_float F, ref int ldf,
         ref float dif,
         ref float scale,
         ref complex_float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtgsyl_")]
        public static extern void DTGSYL(
         ref char trans,
         ref int ijob, ref int m, ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double C, ref int ldc,
         ref double D, ref int ldd,
         ref double E, ref int lde,
         ref double F, ref int ldf,
         ref double dif,
         ref double scale,
         ref double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stgsyl_")]
        public static extern void STGSYL(
         ref char trans,
         ref int ijob, ref int m, ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float C, ref int ldc,
         ref float D, ref int ldd,
         ref float E, ref int lde,
         ref float F, ref int ldf,
         ref float dif,
         ref float scale,
         ref float work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztgsyl_")]
        public static extern void ZTGSYL(
         ref char trans,
         ref int ijob, ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double C, ref int ldc,
         ref complex_double D, ref int ldd,
         ref complex_double E, ref int lde,
         ref complex_double F, ref int ldf,
         ref double dif,
         ref double scale,
         ref complex_double work, ref int lwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctpcon_")]
        public static extern void CTPCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref complex_float AP,
         ref float rcond,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtpcon_")]
        public static extern void DTPCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref double AP,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stpcon_")]
        public static extern void STPCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref float AP,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztpcon_")]
        public static extern void ZTPCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref complex_double AP,
         ref double rcond,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctplqt_")]
        public static extern void CTPLQT(
         ref int m, ref int n, ref int l, ref int mb,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float T, ref int ldt,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtplqt_")]
        public static extern void DTPLQT(
         ref int m, ref int n, ref int l, ref int mb,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double T, ref int ldt,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stplqt_")]
        public static extern void STPLQT(
         ref int m, ref int n, ref int l, ref int mb,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float T, ref int ldt,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztplqt_")]
        public static extern void ZTPLQT(
         ref int m, ref int n, ref int l, ref int mb,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double T, ref int ldt,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctplqt2_")]
        public static extern void CTPLQT2(
         ref int m, ref int n, ref int l,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtplqt2_")]
        public static extern void DTPLQT2(
         ref int m, ref int n, ref int l,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stplqt2_")]
        public static extern void STPLQT2(
         ref int m, ref int n, ref int l,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztplqt2_")]
        public static extern void ZTPLQT2(
         ref int m, ref int n, ref int l,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctpmlqt_")]
        public static extern void CTPMLQT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l, ref int mb,
         ref complex_float V, ref int ldv,
         ref complex_float T, ref int ldt,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtpmlqt_")]
        public static extern void DTPMLQT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l, ref int mb,
         ref double V, ref int ldv,
         ref double T, ref int ldt,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stpmlqt_")]
        public static extern void STPMLQT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l, ref int mb,
         ref float V, ref int ldv,
         ref float T, ref int ldt,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztpmlqt_")]
        public static extern void ZTPMLQT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l, ref int mb,
         ref complex_double V, ref int ldv,
         ref complex_double T, ref int ldt,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctpmqrt_")]
        public static extern void CTPMQRT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l, ref int nb,
         ref complex_float V, ref int ldv,
         ref complex_float T, ref int ldt,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtpmqrt_")]
        public static extern void DTPMQRT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l, ref int nb,
         ref double V, ref int ldv,
         ref double T, ref int ldt,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stpmqrt_")]
        public static extern void STPMQRT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l, ref int nb,
         ref float V, ref int ldv,
         ref float T, ref int ldt,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztpmqrt_")]
        public static extern void ZTPMQRT(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l, ref int nb,
         ref complex_double V, ref int ldv,
         ref complex_double T, ref int ldt,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctpqrt_")]
        public static extern void CTPQRT(
         ref int m, ref int n, ref int l, ref int nb,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float T, ref int ldt,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtpqrt_")]
        public static extern void DTPQRT(
         ref int m, ref int n, ref int l, ref int nb,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double T, ref int ldt,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stpqrt_")]
        public static extern void STPQRT(
         ref int m, ref int n, ref int l, ref int nb,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float T, ref int ldt,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztpqrt_")]
        public static extern void ZTPQRT(
         ref int m, ref int n, ref int l, ref int nb,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double T, ref int ldt,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctpqrt2_")]
        public static extern void CTPQRT2(
         ref int m, ref int n, ref int l,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtpqrt2_")]
        public static extern void DTPQRT2(
         ref int m, ref int n, ref int l,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stpqrt2_")]
        public static extern void STPQRT2(
         ref int m, ref int n, ref int l,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztpqrt2_")]
        public static extern void ZTPQRT2(
         ref int m, ref int n, ref int l,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double T, ref int ldt,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctprfb_")]
        public static extern void CTPRFB(
         ref char side, ref char trans, ref char direct, ref char storev,
         ref int m, ref int n, ref int k, ref int l,
         ref complex_float V, ref int ldv,
         ref complex_float T, ref int ldt,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float work, ref int ldwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtprfb_")]
        public static extern void DTPRFB(
         ref char side, ref char trans, ref char direct, ref char storev,
         ref int m, ref int n, ref int k, ref int l,
         ref double V, ref int ldv,
         ref double T, ref int ldt,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double work, ref int ldwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stprfb_")]
        public static extern void STPRFB(
         ref char side, ref char trans, ref char direct, ref char storev,
         ref int m, ref int n, ref int k, ref int l,
         ref float V, ref int ldv,
         ref float T, ref int ldt,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float work, ref int ldwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztprfb_")]
        public static extern void ZTPRFB(
         ref char side, ref char trans, ref char direct, ref char storev,
         ref int m, ref int n, ref int k, ref int l,
         ref complex_double V, ref int ldv,
         ref complex_double T, ref int ldt,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double work, ref int ldwork);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctprfs_")]
        public static extern void CTPRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtprfs_")]
        public static extern void DTPRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref double AP,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stprfs_")]
        public static extern void STPRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref float AP,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztprfs_")]
        public static extern void ZTPRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctptri_")]
        public static extern void CTPTRI(
         ref char uplo, ref char diag,
         ref int n,
         ref complex_float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtptri_")]
        public static extern void DTPTRI(
         ref char uplo, ref char diag,
         ref int n,
         ref double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stptri_")]
        public static extern void STPTRI(
         ref char uplo, ref char diag,
         ref int n,
         ref float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztptri_")]
        public static extern void ZTPTRI(
         ref char uplo, ref char diag,
         ref int n,
         ref complex_double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctptrs_")]
        public static extern void CTPTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref complex_float AP,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtptrs_")]
        public static extern void DTPTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref double AP,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stptrs_")]
        public static extern void STPTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref float AP,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztptrs_")]
        public static extern void ZTPTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref complex_double AP,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctpttf_")]
        public static extern void CTPTTF(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_float AP,
         ref complex_float ARF,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtpttf_")]
        public static extern void DTPTTF(
         ref char transr, ref char uplo,
         ref int n,
         ref double AP,
         ref double ARF,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stpttf_")]
        public static extern void STPTTF(
         ref char transr, ref char uplo,
         ref int n,
         ref float AP,
         ref float ARF,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztpttf_")]
        public static extern void ZTPTTF(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_double AP,
         ref complex_double ARF,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctpttr_")]
        public static extern void CTPTTR(
         ref char uplo,
         ref int n,
         ref complex_float AP,
         ref complex_float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtpttr_")]
        public static extern void DTPTTR(
         ref char uplo,
         ref int n,
         ref double AP,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stpttr_")]
        public static extern void STPTTR(
         ref char uplo,
         ref int n,
         ref float AP,
         ref float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztpttr_")]
        public static extern void ZTPTTR(
         ref char uplo,
         ref int n,
         ref complex_double AP,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrcon_")]
        public static extern void CTRCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref complex_float A, ref int lda,
         ref float rcond,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrcon_")]
        public static extern void DTRCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref double A, ref int lda,
         ref double rcond,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strcon_")]
        public static extern void STRCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref float A, ref int lda,
         ref float rcond,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrcon_")]
        public static extern void ZTRCON(
         ref char norm, ref char uplo, ref char diag,
         ref int n,
         ref complex_double A, ref int lda,
         ref double rcond,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrevc_")]
        public static extern void CTREVC(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref complex_float T, ref int ldt,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr, ref int mm, ref int m,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrevc_")]
        public static extern void DTREVC(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref double T, ref int ldt,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr, ref int mm, ref int m,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strevc_")]
        public static extern void STREVC(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref float T, ref int ldt,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr, ref int mm, ref int m,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrevc_")]
        public static extern void ZTREVC(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref complex_double T, ref int ldt,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr, ref int mm, ref int m,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrevc3_")]
        public static extern void CTREVC3(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref complex_float T, ref int ldt,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr, ref int mm, ref int m,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrevc3_")]
        public static extern void DTREVC3(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref double T, ref int ldt,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr, ref int mm, ref int m,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strevc3_")]
        public static extern void STREVC3(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref float T, ref int ldt,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr, ref int mm, ref int m,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrevc3_")]
        public static extern void ZTREVC3(
         ref char side, ref char howmny,
         ref int select,
         ref int n,
         ref complex_double T, ref int ldt,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr, ref int mm, ref int m,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrexc_")]
        public static extern void CTREXC(
         ref char compq,
         ref int n,
         ref complex_float T, ref int ldt,
         ref complex_float Q, ref int ldq, ref int ifst, ref int ilst,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrexc_")]
        public static extern void DTREXC(
         ref char compq,
         ref int n,
         ref double T, ref int ldt,
         ref double Q, ref int ldq, ref int ifst, ref int ilst,
         ref double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strexc_")]
        public static extern void STREXC(
         ref char compq,
         ref int n,
         ref float T, ref int ldt,
         ref float Q, ref int ldq, ref int ifst, ref int ilst,
         ref float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrexc_")]
        public static extern void ZTREXC(
         ref char compq,
         ref int n,
         ref complex_double T, ref int ldt,
         ref complex_double Q, ref int ldq, ref int ifst, ref int ilst,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrrfs_")]
        public static extern void CTRRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref complex_float work,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrrfs_")]
        public static extern void DTRRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref double work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strrfs_")]
        public static extern void STRRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float X, ref int ldx,
         ref float ferr,
         ref float berr,
         ref float work,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrrfs_")]
        public static extern void ZTRRFS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double X, ref int ldx,
         ref double ferr,
         ref double berr,
         ref complex_double work,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrsen_")]
        public static extern void CTRSEN(
         ref char job, ref char compq,
         ref int select,
         ref int n,
         ref complex_float T, ref int ldt,
         ref complex_float Q, ref int ldq,
         ref complex_float W, ref int m,
         ref float s,
         ref float sep,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrsen_")]
        public static extern void DTRSEN(
         ref char job, ref char compq,
         ref int select,
         ref int n,
         ref double T, ref int ldt,
         ref double Q, ref int ldq,
         ref double WR,
         ref double WI, ref int m,
         ref double s,
         ref double sep,
         ref double work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strsen_")]
        public static extern void STRSEN(
         ref char job, ref char compq,
         ref int select,
         ref int n,
         ref float T, ref int ldt,
         ref float Q, ref int ldq,
         ref float WR,
         ref float WI, ref int m,
         ref float s,
         ref float sep,
         ref float work, ref int lwork,
         ref int iwork, ref int liwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrsen_")]
        public static extern void ZTRSEN(
         ref char job, ref char compq,
         ref int select,
         ref int n,
         ref complex_double T, ref int ldt,
         ref complex_double Q, ref int ldq,
         ref complex_double W, ref int m,
         ref double s,
         ref double sep,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrsna_")]
        public static extern void CTRSNA(
         ref char job, ref char howmny,
         ref int select,
         ref int n,
         ref complex_float T, ref int ldt,
         ref complex_float VL, ref int ldvl,
         ref complex_float VR, ref int ldvr,
         ref float S,
         ref float SEP, ref int mm, ref int m,
         ref complex_float work, ref int ldwork,
         ref float rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrsna_")]
        public static extern void DTRSNA(
         ref char job, ref char howmny,
         ref int select,
         ref int n,
         ref double T, ref int ldt,
         ref double VL, ref int ldvl,
         ref double VR, ref int ldvr,
         ref double S,
         ref double SEP, ref int mm, ref int m,
         ref double work, ref int ldwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strsna_")]
        public static extern void STRSNA(
         ref char job, ref char howmny,
         ref int select,
         ref int n,
         ref float T, ref int ldt,
         ref float VL, ref int ldvl,
         ref float VR, ref int ldvr,
         ref float S,
         ref float SEP, ref int mm, ref int m,
         ref float work, ref int ldwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrsna_")]
        public static extern void ZTRSNA(
         ref char job, ref char howmny,
         ref int select,
         ref int n,
         ref complex_double T, ref int ldt,
         ref complex_double VL, ref int ldvl,
         ref complex_double VR, ref int ldvr,
         ref double S,
         ref double SEP, ref int mm, ref int m,
         ref complex_double work, ref int ldwork,
         ref double rwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrsyl_")]
        public static extern void CTRSYL(
         ref char trana, ref char tranb,
         ref int isgn, ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref complex_float C, ref int ldc,
         ref float scale,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrsyl_")]
        public static extern void DTRSYL(
         ref char trana, ref char tranb,
         ref int isgn, ref int m, ref int n,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref double C, ref int ldc,
         ref double scale,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strsyl_")]
        public static extern void STRSYL(
         ref char trana, ref char tranb,
         ref int isgn, ref int m, ref int n,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref float C, ref int ldc,
         ref float scale,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrsyl_")]
        public static extern void ZTRSYL(
         ref char trana, ref char tranb,
         ref int isgn, ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref complex_double C, ref int ldc,
         ref double scale,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrtri_")]
        public static extern void CTRTRI(
         ref char uplo, ref char diag,
         ref int n,
         ref complex_float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrtri_")]
        public static extern void DTRTRI(
         ref char uplo, ref char diag,
         ref int n,
         ref double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strtri_")]
        public static extern void STRTRI(
         ref char uplo, ref char diag,
         ref int n,
         ref float A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrtri_")]
        public static extern void ZTRTRI(
         ref char uplo, ref char diag,
         ref int n,
         ref complex_double A, ref int lda,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrtrs_")]
        public static extern void CTRTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref complex_float A, ref int lda,
         ref complex_float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrtrs_")]
        public static extern void DTRTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref double A, ref int lda,
         ref double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strtrs_")]
        public static extern void STRTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref float A, ref int lda,
         ref float B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrtrs_")]
        public static extern void ZTRTRS(
         ref char uplo, ref char trans, ref char diag,
         ref int n, ref int nrhs,
         ref complex_double A, ref int lda,
         ref complex_double B, ref int ldb,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrttf_")]
        public static extern void CTRTTF(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float ARF,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrttf_")]
        public static extern void DTRTTF(
         ref char transr, ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double ARF,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strttf_")]
        public static extern void STRTTF(
         ref char transr, ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float ARF,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrttf_")]
        public static extern void ZTRTTF(
         ref char transr, ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double ARF,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctrttp_")]
        public static extern void CTRTTP(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtrttp_")]
        public static extern void DTRTTP(
         ref char uplo,
         ref int n,
         ref double A, ref int lda,
         ref double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "strttp_")]
        public static extern void STRTTP(
         ref char uplo,
         ref int n,
         ref float A, ref int lda,
         ref float AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztrttp_")]
        public static extern void ZTRTTP(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double AP,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ctzrzf_")]
        public static extern void CTZRZF(
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "dtzrzf_")]
        public static extern void DTZRZF(
         ref int m, ref int n,
         ref double A, ref int lda,
         ref double tau,
         ref double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "stzrzf_")]
        public static extern void STZRZF(
         ref int m, ref int n,
         ref float A, ref int lda,
         ref float tau,
         ref float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "ztzrzf_")]
        public static extern void ZTZRZF(
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunbdb_")]
        public static extern void CUNBDB(
         ref char trans, ref char signs,
         ref int m, ref int p, ref int q,
         ref complex_float X11, ref int ldx11,
         ref complex_float X12, ref int ldx12,
         ref complex_float X21, ref int ldx21,
         ref complex_float X22, ref int ldx22,
         ref float theta,
         ref float phi,
         ref complex_float TAUP1,
         ref complex_float TAUP2,
         ref complex_float TAUQ1,
         ref complex_float TAUQ2,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunbdb_")]
        public static extern void ZUNBDB(
         ref char trans, ref char signs,
         ref int m, ref int p, ref int q,
         ref complex_double X11, ref int ldx11,
         ref complex_double X12, ref int ldx12,
         ref complex_double X21, ref int ldx21,
         ref complex_double X22, ref int ldx22,
         ref double theta,
         ref double phi,
         ref complex_double TAUP1,
         ref complex_double TAUP2,
         ref complex_double TAUQ1,
         ref complex_double TAUQ2,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cuncsd_")]
        public static extern void CUNCSD(
         ref char jobu1, ref char jobu2, ref char jobv1t, ref char jobv2t, ref char trans, ref char signs,
         ref int m, ref int p, ref int q,
         ref complex_float X11, ref int ldx11,
         ref complex_float X12, ref int ldx12,
         ref complex_float X21, ref int ldx21,
         ref complex_float X22, ref int ldx22,
         ref float theta,
         ref complex_float U1, ref int ldu1,
         ref complex_float U2, ref int ldu2,
         ref complex_float V1T, ref int ldv1t,
         ref complex_float V2T, ref int ldv2t,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zuncsd_")]
        public static extern void ZUNCSD(
         ref char jobu1, ref char jobu2, ref char jobv1t, ref char jobv2t, ref char trans, ref char signs,
         ref int m, ref int p, ref int q,
         ref complex_double X11, ref int ldx11,
         ref complex_double X12, ref int ldx12,
         ref complex_double X21, ref int ldx21,
         ref complex_double X22, ref int ldx22,
         ref double theta,
         ref complex_double U1, ref int ldu1,
         ref complex_double U2, ref int ldu2,
         ref complex_double V1T, ref int ldv1t,
         ref complex_double V2T, ref int ldv2t,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cuncsd2by1_")]
        public static extern void CUNCSD2BY1(
         ref char jobu1, ref char jobu2, ref char jobv1t,
         ref int m, ref int p, ref int q,
         ref complex_float X11, ref int ldx11,
         ref complex_float X21, ref int ldx21,
         ref float theta,
         ref complex_float U1, ref int ldu1,
         ref complex_float U2, ref int ldu2,
         ref complex_float V1T, ref int ldv1t,
         ref complex_float work, ref int lwork,
         ref float rwork, ref int lrwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zuncsd2by1_")]
        public static extern void ZUNCSD2BY1(
         ref char jobu1, ref char jobu2, ref char jobv1t,
         ref int m, ref int p, ref int q,
         ref complex_double X11, ref int ldx11,
         ref complex_double X21, ref int ldx21,
         ref double theta,
         ref complex_double U1, ref int ldu1,
         ref complex_double U2, ref int ldu2,
         ref complex_double V1T, ref int ldv1t,
         ref complex_double work, ref int lwork,
         ref double rwork, ref int lrwork,
         ref int iwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cungbr_")]
        public static extern void CUNGBR(
         ref char vect,
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zungbr_")]
        public static extern void ZUNGBR(
         ref char vect,
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunghr_")]
        public static extern void CUNGHR(
         ref int n, ref int ilo, ref int ihi,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunghr_")]
        public static extern void ZUNGHR(
         ref int n, ref int ilo, ref int ihi,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunglq_")]
        public static extern void CUNGLQ(
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunglq_")]
        public static extern void ZUNGLQ(
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cungql_")]
        public static extern void CUNGQL(
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zungql_")]
        public static extern void ZUNGQL(
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cungqr_")]
        public static extern void CUNGQR(
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zungqr_")]
        public static extern void ZUNGQR(
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cungrq_")]
        public static extern void CUNGRQ(
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zungrq_")]
        public static extern void ZUNGRQ(
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cungtr_")]
        public static extern void CUNGTR(
         ref char uplo,
         ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zungtr_")]
        public static extern void ZUNGTR(
         ref char uplo,
         ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunmbr_")]
        public static extern void CUNMBR(
         ref char vect, ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunmbr_")]
        public static extern void ZUNMBR(
         ref char vect, ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunmhr_")]
        public static extern void CUNMHR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int ilo, ref int ihi,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunmhr_")]
        public static extern void ZUNMHR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int ilo, ref int ihi,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunmlq_")]
        public static extern void CUNMLQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunmlq_")]
        public static extern void ZUNMLQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunmql_")]
        public static extern void CUNMQL(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunmql_")]
        public static extern void ZUNMQL(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunmqr_")]
        public static extern void CUNMQR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunmqr_")]
        public static extern void ZUNMQR(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunmrq_")]
        public static extern void CUNMRQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunmrq_")]
        public static extern void ZUNMRQ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunmrz_")]
        public static extern void CUNMRZ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunmrz_")]
        public static extern void ZUNMRZ(
         ref char side, ref char trans,
         ref int m, ref int n, ref int k, ref int l,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cunmtr_")]
        public static extern void CUNMTR(
         ref char side, ref char uplo, ref char trans,
         ref int m, ref int n,
         ref complex_float A, ref int lda,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zunmtr_")]
        public static extern void ZUNMTR(
         ref char side, ref char uplo, ref char trans,
         ref int m, ref int n,
         ref complex_double A, ref int lda,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work, ref int lwork,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cupgtr_")]
        public static extern void CUPGTR(
         ref char uplo,
         ref int n,
         ref complex_float AP,
         ref complex_float tau,
         ref complex_float Q, ref int ldq,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zupgtr_")]
        public static extern void ZUPGTR(
         ref char uplo,
         ref int n,
         ref complex_double AP,
         ref complex_double tau,
         ref complex_double Q, ref int ldq,
         ref complex_double work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "cupmtr_")]
        public static extern void CUPMTR(
         ref char side, ref char uplo, ref char trans,
         ref int m, ref int n,
         ref complex_float AP,
         ref complex_float tau,
         ref complex_float C, ref int ldc,
         ref complex_float work,
         ref int info);

        [DllImport("openblas.dll", CharSet = CharSet.Ansi, EntryPoint = "zupmtr_")]
        public static extern void ZUPMTR(
         ref char side, ref char uplo, ref char trans,
         ref int m, ref int n,
         ref complex_double AP,
         ref complex_double tau,
         ref complex_double C, ref int ldc,
         ref complex_double work,
         ref int info);
    }
}
