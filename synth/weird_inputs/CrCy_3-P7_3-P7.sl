;synthesize a countermeasure for the following program
;bool compute( bool k1, bool k2, bool k3, bool r1, bool r2, bool r3, bool r4){
;bool n00;
;bool n01;
;bool n02;
;bool n03;
;bool n04;
;bool n05;
;bool n06;
;bool n07;
;bool n08;
;bool n09;
;bool n10;
;bool n11;
;bool n12;
;bool n13;
;bool n14;
;bool n15;
;bool n16;
;bool n17;
;bool n18;
;bool n19;
;bool n20;
;bool n21;
;bool n22;
;bool n23;
;bool n24;
;bool n25;
;bool n26;
;bool n27;
;bool n28;
;bool n29;
;bool n30;
;bool n31;
;bool n32;
;
;
; n31 = r3 ^ k3 ;
; n29 = r1 ;
; n28 = r2 ^ k2 ;
; n27 = r3 ^ k3 ;
; n25 = r1 ;
; n24 = r2 ;
; n21 = r2 ^ k2 ;
; n20 = r3 ;
; n19 = r1 ^ k1 ;
; n18 = r3 ^ k3 ;
; n17 = r3 ;
; n16 = r2 ;
; n15 = 1 ^ n31 ;
; n14 = n28 ^ r1 ;
; n13 = 0 ^ n27 ;
; n12 = r2 ^ r1 ;
; n11 = ~ 0 ;
; n10 = r3 & n21 ;
; n09 = n18 ^ n19 ;
; n08 = r2 | r3 ;
; n07 = n14 | n15 ;
; n06 = n12 | n13 ;
; n05 = n10 & n11 ;
; n04 =  n08 ^  n09 ;
; n03 =  n06 &  n07 ;
; n02 =  n04 ^  n03 ;
; n01 =  n02 ^  r4 ;
; n00 =  n01 ^ n05 ;
; return( n00) ;
;}

(set-logic BV)

(define-fun Spec ((k1 Bool) (k2 Bool) (k3 Bool) (r1 Bool) (r2 Bool) (r3 Bool) (r4 Bool)) Bool
  (xor (xor (xor (xor (or r2 r3) (xor (xor r3 k3) (xor r1 k1))) (and (or (xor r2 r1) (xor (xor r3 k3) false)) (or (xor r1 (xor r2 k2)) (xor (xor r3 k3) true)))) r4) (and (and r3 (xor r2 k2)) (not false)))
)
 
(synth-fun Imp ((k1 Bool) (k2 Bool) (k3 Bool) (r1 Bool) (r2 Bool) (r3 Bool) (r4 Bool)) Bool
 ((Start Bool ( (and depth1 depth1) (or depth1 depth1) (xor depth1 depth1) (not depth1) ) )
 (depth1 Bool ( (and depth2 depth2) (or depth2 depth2) (xor depth2 depth2) (not depth2) ) )
 (depth2 Bool ( (and depth3 depth3) (or depth3 depth3) (xor depth3 depth3) (not depth3) ) )
 (depth3 Bool ( (and depth4 depth4) (or depth4 depth4) (xor depth4 depth4) (not depth4) ) )
 (depth4 Bool ( (and depth5 depth5) (or depth5 depth5) (xor depth5 depth5) (not depth5) ) )
 (depth5 Bool ( 
  k1 
  k2 
  k3
  r1 
  r2 
  r3 
  r4 
  ) ) )
)
 
(declare-var k1 Bool)
(declare-var k2 Bool)
(declare-var k3 Bool)
(declare-var r1 Bool)
(declare-var r2 Bool)
(declare-var r3 Bool)
(declare-var r4 Bool)

(constraint (= (Spec k1 k2 k3 r1 r2 r3 r4) (Imp k1 k2 k3 r1 r2 r3 r4)))
(check-synth)

