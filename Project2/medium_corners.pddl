(define (problem medium_corners)
	(:domain pacman)
	(:objects
		 sq0_23 sq0_22 sq0_21 sq0_20 sq0_27 sq9_21 sq0_25 sq0_24 sq9_27 sq0_0 sq0_1 sq0_2 sq0_3 sq0_4 sq0_5 sq0_6 sq0_8 sq2_6 sq2_4 sq2_5 sq2_2 sq2_3 sq2_0 sq2_1 sq5_19 sq2_8 sq2_9 sq11_4 sq11_1 sq11_0 sq11_2 sq7_23 sq7_21 sq7_27 sq7_25 sq8_8 sq8_9 sq8_0 sq8_2 sq8_3 sq8_4 sq8_5 sq8_6 sq6_11 sq8_12 sq10_17 sq8_10 sq8_11 sq8_16 sq8_17 sq10_10 sq8_15 sq4_27 sq4_26 sq4_25 sq8_19 sq4_23 sq4_22 sq4_21 sq4_20 sq4_4 sq10_12 sq4_6 sq4_0 sq4_1 sq4_2 sq4_8 sq8_7 sq11_25 sq10_13 sq6_8 sq6_9 sq9_25 sq6_2 sq6_3 sq6_0 sq6_1 sq6_6 sq10_14 sq6_4 sq6_5 sq1_8 sq10_15 sq1_1 sq1_0 sq1_3 sq1_2 sq1_5 sq1_4 sq1_6 sq7_2 sq10_11 sq11_27 sq10_8 sq8_18 sq2_25 sq9_0 sq2_27 sq2_26 sq2_21 sq2_20 sq2_23 sq9_8 sq7_4 sq10_4 sq10_5 sq10_19 sq11_24 sq7_12 sq11_26 sq10_7 sq7_17 sq11_21 sq11_23 sq3_4 sq10_1 sq5_8 sq5_4 sq5_6 sq5_0 sq2_10 sq2_11 sq2_12 sq2_13 sq2_14 sq2_15 sq2_17 sq2_18 sq2_19 sq10_6 sq3_25 sq10_0 sq3_23 sq10_2 sq8_23 sq8_22 sq8_21 sq8_27 sq8_25 sq8_24 sq11_11 sq1_10 sq11_13 sq11_12 sq11_15 sq11_14 sq1_17 sq10_27 sq10_23 sq10_21 sq10_20 sq0_18 sq0_19 sq0_12 sq0_13 sq0_10 sq6_18 sq0_16 sq0_17 sq0_14 sq0_15 sq6_21 sq6_23 sq6_25 sq6_27 sq6_26 sq11_10 sq1_25 sq1_27 sq11_17 sq4_16 sq4_17 sq4_14 sq4_15 sq4_12 sq6_19 sq4_10 sq4_11 sq6_15 sq6_16 sq6_17 sq6_10 sq5_10 sq6_12 sq4_19 sq3_17 sq3_14 sq3_10 sq9_10 sq9_15 sq9_17 sq9_2 sq5_23 sq5_27 - position
	)
	(:init
		(pacman-at-position sq11_4)
		(empty-at-position sq0_23)
		(empty-at-position sq0_22)
		(empty-at-position sq0_21)
		(empty-at-position sq0_20)
		(empty-at-position sq9_21)
		(empty-at-position sq0_25)
		(empty-at-position sq0_24)
		(empty-at-position sq9_27)
		(empty-at-position sq0_1)
		(empty-at-position sq0_2)
		(empty-at-position sq0_3)
		(empty-at-position sq0_4)
		(empty-at-position sq0_5)
		(empty-at-position sq0_6)
		(empty-at-position sq0_8)
		(empty-at-position sq2_6)
		(empty-at-position sq2_4)
		(empty-at-position sq2_5)
		(empty-at-position sq2_2)
		(empty-at-position sq2_3)
		(empty-at-position sq2_0)
		(empty-at-position sq2_1)
		(empty-at-position sq5_19)
		(empty-at-position sq2_8)
		(empty-at-position sq2_9)
		(empty-at-position sq11_4)
		(empty-at-position sq11_1)
		(empty-at-position sq11_2)
		(empty-at-position sq7_23)
		(empty-at-position sq7_21)
		(empty-at-position sq7_27)
		(empty-at-position sq7_25)
		(empty-at-position sq8_8)
		(empty-at-position sq8_9)
		(empty-at-position sq8_0)
		(empty-at-position sq8_2)
		(empty-at-position sq8_3)
		(empty-at-position sq8_4)
		(empty-at-position sq8_5)
		(empty-at-position sq8_6)
		(empty-at-position sq6_11)
		(empty-at-position sq8_12)
		(empty-at-position sq10_17)
		(empty-at-position sq8_10)
		(empty-at-position sq8_11)
		(empty-at-position sq8_16)
		(empty-at-position sq8_17)
		(empty-at-position sq10_10)
		(empty-at-position sq8_15)
		(empty-at-position sq4_27)
		(empty-at-position sq4_26)
		(empty-at-position sq4_25)
		(empty-at-position sq8_19)
		(empty-at-position sq4_23)
		(empty-at-position sq4_22)
		(empty-at-position sq4_21)
		(empty-at-position sq4_20)
		(empty-at-position sq4_4)
		(empty-at-position sq10_12)
		(empty-at-position sq4_6)
		(empty-at-position sq4_0)
		(empty-at-position sq4_1)
		(empty-at-position sq4_2)
		(empty-at-position sq4_8)
		(empty-at-position sq8_7)
		(empty-at-position sq11_25)
		(empty-at-position sq10_13)
		(empty-at-position sq6_8)
		(empty-at-position sq6_9)
		(empty-at-position sq9_25)
		(empty-at-position sq6_2)
		(empty-at-position sq6_3)
		(empty-at-position sq6_0)
		(empty-at-position sq6_1)
		(empty-at-position sq6_6)
		(empty-at-position sq10_14)
		(empty-at-position sq6_4)
		(empty-at-position sq6_5)
		(empty-at-position sq1_8)
		(empty-at-position sq10_15)
		(empty-at-position sq1_1)
		(empty-at-position sq1_0)
		(empty-at-position sq1_3)
		(empty-at-position sq1_2)
		(empty-at-position sq1_5)
		(empty-at-position sq1_4)
		(empty-at-position sq1_6)
		(empty-at-position sq7_2)
		(empty-at-position sq10_11)
		(empty-at-position sq10_8)
		(empty-at-position sq8_18)
		(empty-at-position sq2_25)
		(empty-at-position sq9_0)
		(empty-at-position sq2_27)
		(empty-at-position sq2_26)
		(empty-at-position sq2_21)
		(empty-at-position sq2_20)
		(empty-at-position sq2_23)
		(empty-at-position sq9_8)
		(empty-at-position sq7_4)
		(empty-at-position sq10_4)
		(empty-at-position sq10_5)
		(empty-at-position sq10_19)
		(empty-at-position sq11_24)
		(empty-at-position sq7_12)
		(empty-at-position sq11_26)
		(empty-at-position sq10_7)
		(empty-at-position sq7_17)
		(empty-at-position sq11_21)
		(empty-at-position sq11_23)
		(empty-at-position sq3_4)
		(empty-at-position sq10_1)
		(empty-at-position sq5_8)
		(empty-at-position sq5_4)
		(empty-at-position sq5_6)
		(empty-at-position sq5_0)
		(empty-at-position sq2_10)
		(empty-at-position sq2_11)
		(empty-at-position sq2_12)
		(empty-at-position sq2_13)
		(empty-at-position sq2_14)
		(empty-at-position sq2_15)
		(empty-at-position sq2_17)
		(empty-at-position sq2_18)
		(empty-at-position sq2_19)
		(empty-at-position sq10_6)
		(empty-at-position sq3_25)
		(empty-at-position sq10_0)
		(empty-at-position sq3_23)
		(empty-at-position sq10_2)
		(empty-at-position sq8_23)
		(empty-at-position sq8_22)
		(empty-at-position sq8_21)
		(empty-at-position sq8_27)
		(empty-at-position sq8_25)
		(empty-at-position sq8_24)
		(empty-at-position sq11_11)
		(empty-at-position sq1_10)
		(empty-at-position sq11_13)
		(empty-at-position sq11_12)
		(empty-at-position sq11_15)
		(empty-at-position sq11_14)
		(empty-at-position sq1_17)
		(empty-at-position sq10_27)
		(empty-at-position sq10_23)
		(empty-at-position sq10_21)
		(empty-at-position sq10_20)
		(empty-at-position sq0_18)
		(empty-at-position sq0_19)
		(empty-at-position sq0_12)
		(empty-at-position sq0_13)
		(empty-at-position sq0_10)
		(empty-at-position sq6_18)
		(empty-at-position sq0_16)
		(empty-at-position sq0_17)
		(empty-at-position sq0_14)
		(empty-at-position sq0_15)
		(empty-at-position sq6_21)
		(empty-at-position sq6_23)
		(empty-at-position sq6_25)
		(empty-at-position sq6_27)
		(empty-at-position sq6_26)
		(empty-at-position sq11_10)
		(empty-at-position sq1_25)
		(empty-at-position sq1_27)
		(empty-at-position sq11_17)
		(empty-at-position sq4_16)
		(empty-at-position sq4_17)
		(empty-at-position sq4_14)
		(empty-at-position sq4_15)
		(empty-at-position sq4_12)
		(empty-at-position sq6_19)
		(empty-at-position sq4_10)
		(empty-at-position sq4_11)
		(empty-at-position sq6_15)
		(empty-at-position sq6_16)
		(empty-at-position sq6_17)
		(empty-at-position sq6_10)
		(empty-at-position sq5_10)
		(empty-at-position sq6_12)
		(empty-at-position sq4_19)
		(empty-at-position sq3_17)
		(empty-at-position sq3_14)
		(empty-at-position sq3_10)
		(empty-at-position sq9_10)
		(empty-at-position sq9_15)
		(empty-at-position sq9_17)
		(empty-at-position sq9_2)
		(empty-at-position sq5_23)
		(empty-at-position sq5_27)
		(adjacent sq0_23 sq0_22)
		(adjacent sq0_23 sq0_24)
		(adjacent sq0_22 sq0_21)
		(adjacent sq0_22 sq0_23)
		(adjacent sq0_21 sq0_20)
		(adjacent sq0_21 sq0_22)
		(adjacent sq0_20 sq0_19)
		(adjacent sq0_20 sq0_21)
		(adjacent sq0_27 sq1_27)
		(adjacent sq9_21 sq8_21)
		(adjacent sq9_21 sq10_21)
		(adjacent sq0_25 sq1_25)
		(adjacent sq0_25 sq0_24)
		(adjacent sq0_24 sq0_23)
		(adjacent sq0_24 sq0_25)
		(adjacent sq9_27 sq8_27)
		(adjacent sq9_27 sq10_27)
		(adjacent sq0_0 sq1_0)
		(adjacent sq0_0 sq0_1)
		(adjacent sq0_1 sq1_1)
		(adjacent sq0_1 sq0_0)
		(adjacent sq0_1 sq0_2)
		(adjacent sq0_2 sq1_2)
		(adjacent sq0_2 sq0_1)
		(adjacent sq0_2 sq0_3)
		(adjacent sq0_3 sq1_3)
		(adjacent sq0_3 sq0_2)
		(adjacent sq0_3 sq0_4)
		(adjacent sq0_4 sq1_4)
		(adjacent sq0_4 sq0_3)
		(adjacent sq0_4 sq0_5)
		(adjacent sq0_5 sq1_5)
		(adjacent sq0_5 sq0_4)
		(adjacent sq0_5 sq0_6)
		(adjacent sq0_6 sq1_6)
		(adjacent sq0_6 sq0_5)
		(adjacent sq0_8 sq1_8)
		(adjacent sq2_6 sq1_6)
		(adjacent sq2_6 sq2_5)
		(adjacent sq2_4 sq1_4)
		(adjacent sq2_4 sq3_4)
		(adjacent sq2_4 sq2_3)
		(adjacent sq2_4 sq2_5)
		(adjacent sq2_5 sq1_5)
		(adjacent sq2_5 sq2_4)
		(adjacent sq2_5 sq2_6)
		(adjacent sq2_2 sq1_2)
		(adjacent sq2_2 sq2_1)
		(adjacent sq2_2 sq2_3)
		(adjacent sq2_3 sq1_3)
		(adjacent sq2_3 sq2_2)
		(adjacent sq2_3 sq2_4)
		(adjacent sq2_0 sq1_0)
		(adjacent sq2_0 sq2_1)
		(adjacent sq2_1 sq1_1)
		(adjacent sq2_1 sq2_0)
		(adjacent sq2_1 sq2_2)
		(adjacent sq5_19 sq4_19)
		(adjacent sq5_19 sq6_19)
		(adjacent sq2_8 sq1_8)
		(adjacent sq2_8 sq2_9)
		(adjacent sq2_9 sq2_8)
		(adjacent sq2_9 sq2_10)
		(adjacent sq11_4 sq10_4)
		(adjacent sq11_1 sq10_1)
		(adjacent sq11_1 sq11_0)
		(adjacent sq11_1 sq11_2)
		(adjacent sq11_0 sq10_0)
		(adjacent sq11_0 sq11_1)
		(adjacent sq11_2 sq10_2)
		(adjacent sq11_2 sq11_1)
		(adjacent sq7_23 sq6_23)
		(adjacent sq7_23 sq8_23)
		(adjacent sq7_21 sq6_21)
		(adjacent sq7_21 sq8_21)
		(adjacent sq7_27 sq6_27)
		(adjacent sq7_27 sq8_27)
		(adjacent sq7_25 sq6_25)
		(adjacent sq7_25 sq8_25)
		(adjacent sq8_8 sq9_8)
		(adjacent sq8_8 sq8_7)
		(adjacent sq8_8 sq8_9)
		(adjacent sq8_9 sq8_8)
		(adjacent sq8_9 sq8_10)
		(adjacent sq8_0 sq9_0)
		(adjacent sq8_2 sq7_2)
		(adjacent sq8_2 sq9_2)
		(adjacent sq8_2 sq8_3)
		(adjacent sq8_3 sq8_2)
		(adjacent sq8_3 sq8_4)
		(adjacent sq8_4 sq7_4)
		(adjacent sq8_4 sq8_3)
		(adjacent sq8_4 sq8_5)
		(adjacent sq8_5 sq8_4)
		(adjacent sq8_5 sq8_6)
		(adjacent sq8_6 sq8_5)
		(adjacent sq8_6 sq8_7)
		(adjacent sq6_11 sq6_10)
		(adjacent sq6_11 sq6_12)
		(adjacent sq8_12 sq7_12)
		(adjacent sq8_12 sq8_11)
		(adjacent sq10_17 sq9_17)
		(adjacent sq10_17 sq11_17)
		(adjacent sq8_10 sq9_10)
		(adjacent sq8_10 sq8_9)
		(adjacent sq8_10 sq8_11)
		(adjacent sq8_11 sq8_10)
		(adjacent sq8_11 sq8_12)
		(adjacent sq8_16 sq8_15)
		(adjacent sq8_16 sq8_17)
		(adjacent sq8_17 sq7_17)
		(adjacent sq8_17 sq9_17)
		(adjacent sq8_17 sq8_16)
		(adjacent sq8_17 sq8_18)
		(adjacent sq10_10 sq9_10)
		(adjacent sq10_10 sq11_10)
		(adjacent sq10_10 sq10_11)
		(adjacent sq8_15 sq9_15)
		(adjacent sq8_15 sq8_16)
		(adjacent sq4_27 sq5_27)
		(adjacent sq4_27 sq4_26)
		(adjacent sq4_26 sq4_25)
		(adjacent sq4_26 sq4_27)
		(adjacent sq4_25 sq3_25)
		(adjacent sq4_25 sq4_26)
		(adjacent sq8_19 sq8_18)
		(adjacent sq4_23 sq3_23)
		(adjacent sq4_23 sq5_23)
		(adjacent sq4_23 sq4_22)
		(adjacent sq4_22 sq4_21)
		(adjacent sq4_22 sq4_23)
		(adjacent sq4_21 sq4_20)
		(adjacent sq4_21 sq4_22)
		(adjacent sq4_20 sq4_19)
		(adjacent sq4_20 sq4_21)
		(adjacent sq4_4 sq3_4)
		(adjacent sq4_4 sq5_4)
		(adjacent sq10_12 sq11_12)
		(adjacent sq10_12 sq10_11)
		(adjacent sq10_12 sq10_13)
		(adjacent sq4_6 sq5_6)
		(adjacent sq4_0 sq5_0)
		(adjacent sq4_0 sq4_1)
		(adjacent sq4_1 sq4_0)
		(adjacent sq4_1 sq4_2)
		(adjacent sq4_2 sq4_1)
		(adjacent sq4_8 sq5_8)
		(adjacent sq8_7 sq8_6)
		(adjacent sq8_7 sq8_8)
		(adjacent sq11_25 sq11_24)
		(adjacent sq11_25 sq11_26)
		(adjacent sq10_13 sq11_13)
		(adjacent sq10_13 sq10_12)
		(adjacent sq10_13 sq10_14)
		(adjacent sq6_8 sq5_8)
		(adjacent sq6_8 sq6_9)
		(adjacent sq6_9 sq6_8)
		(adjacent sq6_9 sq6_10)
		(adjacent sq9_25 sq8_25)
		(adjacent sq6_2 sq7_2)
		(adjacent sq6_2 sq6_1)
		(adjacent sq6_2 sq6_3)
		(adjacent sq6_3 sq6_2)
		(adjacent sq6_3 sq6_4)
		(adjacent sq6_0 sq5_0)
		(adjacent sq6_0 sq6_1)
		(adjacent sq6_1 sq6_0)
		(adjacent sq6_1 sq6_2)
		(adjacent sq6_6 sq5_6)
		(adjacent sq6_6 sq6_5)
		(adjacent sq10_14 sq11_14)
		(adjacent sq10_14 sq10_13)
		(adjacent sq10_14 sq10_15)
		(adjacent sq6_4 sq5_4)
		(adjacent sq6_4 sq7_4)
		(adjacent sq6_4 sq6_3)
		(adjacent sq6_4 sq6_5)
		(adjacent sq6_5 sq6_4)
		(adjacent sq6_5 sq6_6)
		(adjacent sq1_8 sq0_8)
		(adjacent sq1_8 sq2_8)
		(adjacent sq10_15 sq9_15)
		(adjacent sq10_15 sq11_15)
		(adjacent sq10_15 sq10_14)
		(adjacent sq1_1 sq0_1)
		(adjacent sq1_1 sq2_1)
		(adjacent sq1_1 sq1_0)
		(adjacent sq1_1 sq1_2)
		(adjacent sq1_0 sq0_0)
		(adjacent sq1_0 sq2_0)
		(adjacent sq1_0 sq1_1)
		(adjacent sq1_3 sq0_3)
		(adjacent sq1_3 sq2_3)
		(adjacent sq1_3 sq1_2)
		(adjacent sq1_3 sq1_4)
		(adjacent sq1_2 sq0_2)
		(adjacent sq1_2 sq2_2)
		(adjacent sq1_2 sq1_1)
		(adjacent sq1_2 sq1_3)
		(adjacent sq1_5 sq0_5)
		(adjacent sq1_5 sq2_5)
		(adjacent sq1_5 sq1_4)
		(adjacent sq1_5 sq1_6)
		(adjacent sq1_4 sq0_4)
		(adjacent sq1_4 sq2_4)
		(adjacent sq1_4 sq1_3)
		(adjacent sq1_4 sq1_5)
		(adjacent sq1_6 sq0_6)
		(adjacent sq1_6 sq2_6)
		(adjacent sq1_6 sq1_5)
		(adjacent sq7_2 sq6_2)
		(adjacent sq7_2 sq8_2)
		(adjacent sq10_11 sq11_11)
		(adjacent sq10_11 sq10_10)
		(adjacent sq10_11 sq10_12)
		(adjacent sq11_27 sq10_27)
		(adjacent sq11_27 sq11_26)
		(adjacent sq10_8 sq9_8)
		(adjacent sq10_8 sq10_7)
		(adjacent sq8_18 sq8_17)
		(adjacent sq8_18 sq8_19)
		(adjacent sq2_25 sq1_25)
		(adjacent sq2_25 sq3_25)
		(adjacent sq2_25 sq2_26)
		(adjacent sq9_0 sq8_0)
		(adjacent sq9_0 sq10_0)
		(adjacent sq2_27 sq1_27)
		(adjacent sq2_27 sq2_26)
		(adjacent sq2_26 sq2_25)
		(adjacent sq2_26 sq2_27)
		(adjacent sq2_21 sq2_20)
		(adjacent sq2_20 sq2_19)
		(adjacent sq2_20 sq2_21)
		(adjacent sq2_23 sq3_23)
		(adjacent sq9_8 sq8_8)
		(adjacent sq9_8 sq10_8)
		(adjacent sq7_4 sq6_4)
		(adjacent sq7_4 sq8_4)
		(adjacent sq10_4 sq10_5)
		(adjacent sq10_5 sq10_4)
		(adjacent sq10_5 sq10_6)
		(adjacent sq10_19 sq10_20)
		(adjacent sq11_24 sq11_23)
		(adjacent sq11_24 sq11_25)
		(adjacent sq7_12 sq6_12)
		(adjacent sq7_12 sq8_12)
		(adjacent sq11_26 sq11_25)
		(adjacent sq11_26 sq11_27)
		(adjacent sq10_7 sq10_6)
		(adjacent sq10_7 sq10_8)
		(adjacent sq7_17 sq6_17)
		(adjacent sq7_17 sq8_17)
		(adjacent sq11_21 sq10_21)
		(adjacent sq11_23 sq10_23)
		(adjacent sq11_23 sq11_24)
		(adjacent sq3_4 sq2_4)
		(adjacent sq3_4 sq4_4)
		(adjacent sq10_1 sq11_1)
		(adjacent sq10_1 sq10_0)
		(adjacent sq10_1 sq10_2)
		(adjacent sq5_8 sq4_8)
		(adjacent sq5_8 sq6_8)
		(adjacent sq5_4 sq4_4)
		(adjacent sq5_4 sq6_4)
		(adjacent sq5_6 sq4_6)
		(adjacent sq5_6 sq6_6)
		(adjacent sq5_0 sq4_0)
		(adjacent sq5_0 sq6_0)
		(adjacent sq2_10 sq1_10)
		(adjacent sq2_10 sq3_10)
		(adjacent sq2_10 sq2_9)
		(adjacent sq2_10 sq2_11)
		(adjacent sq2_11 sq2_10)
		(adjacent sq2_11 sq2_12)
		(adjacent sq2_12 sq2_11)
		(adjacent sq2_12 sq2_13)
		(adjacent sq2_13 sq2_12)
		(adjacent sq2_13 sq2_14)
		(adjacent sq2_14 sq3_14)
		(adjacent sq2_14 sq2_13)
		(adjacent sq2_14 sq2_15)
		(adjacent sq2_15 sq2_14)
		(adjacent sq2_17 sq1_17)
		(adjacent sq2_17 sq3_17)
		(adjacent sq2_17 sq2_18)
		(adjacent sq2_18 sq2_17)
		(adjacent sq2_18 sq2_19)
		(adjacent sq2_19 sq2_18)
		(adjacent sq2_19 sq2_20)
		(adjacent sq10_6 sq10_5)
		(adjacent sq10_6 sq10_7)
		(adjacent sq3_25 sq2_25)
		(adjacent sq3_25 sq4_25)
		(adjacent sq10_0 sq9_0)
		(adjacent sq10_0 sq11_0)
		(adjacent sq10_0 sq10_1)
		(adjacent sq3_23 sq2_23)
		(adjacent sq3_23 sq4_23)
		(adjacent sq10_2 sq9_2)
		(adjacent sq10_2 sq11_2)
		(adjacent sq10_2 sq10_1)
		(adjacent sq8_23 sq7_23)
		(adjacent sq8_23 sq8_22)
		(adjacent sq8_23 sq8_24)
		(adjacent sq8_22 sq8_21)
		(adjacent sq8_22 sq8_23)
		(adjacent sq8_21 sq7_21)
		(adjacent sq8_21 sq9_21)
		(adjacent sq8_21 sq8_22)
		(adjacent sq8_27 sq7_27)
		(adjacent sq8_27 sq9_27)
		(adjacent sq8_25 sq7_25)
		(adjacent sq8_25 sq9_25)
		(adjacent sq8_25 sq8_24)
		(adjacent sq8_24 sq8_23)
		(adjacent sq8_24 sq8_25)
		(adjacent sq11_11 sq10_11)
		(adjacent sq11_11 sq11_10)
		(adjacent sq11_11 sq11_12)
		(adjacent sq1_10 sq0_10)
		(adjacent sq1_10 sq2_10)
		(adjacent sq11_13 sq10_13)
		(adjacent sq11_13 sq11_12)
		(adjacent sq11_13 sq11_14)
		(adjacent sq11_12 sq10_12)
		(adjacent sq11_12 sq11_11)
		(adjacent sq11_12 sq11_13)
		(adjacent sq11_15 sq10_15)
		(adjacent sq11_15 sq11_14)
		(adjacent sq11_14 sq10_14)
		(adjacent sq11_14 sq11_13)
		(adjacent sq11_14 sq11_15)
		(adjacent sq1_17 sq0_17)
		(adjacent sq1_17 sq2_17)
		(adjacent sq10_27 sq9_27)
		(adjacent sq10_27 sq11_27)
		(adjacent sq10_23 sq11_23)
		(adjacent sq10_21 sq9_21)
		(adjacent sq10_21 sq11_21)
		(adjacent sq10_21 sq10_20)
		(adjacent sq10_20 sq10_19)
		(adjacent sq10_20 sq10_21)
		(adjacent sq0_18 sq0_17)
		(adjacent sq0_18 sq0_19)
		(adjacent sq0_19 sq0_18)
		(adjacent sq0_19 sq0_20)
		(adjacent sq0_12 sq0_13)
		(adjacent sq0_13 sq0_12)
		(adjacent sq0_13 sq0_14)
		(adjacent sq0_10 sq1_10)
		(adjacent sq6_18 sq6_17)
		(adjacent sq6_18 sq6_19)
		(adjacent sq0_16 sq0_15)
		(adjacent sq0_16 sq0_17)
		(adjacent sq0_17 sq1_17)
		(adjacent sq0_17 sq0_16)
		(adjacent sq0_17 sq0_18)
		(adjacent sq0_14 sq0_13)
		(adjacent sq0_14 sq0_15)
		(adjacent sq0_15 sq0_14)
		(adjacent sq0_15 sq0_16)
		(adjacent sq6_21 sq7_21)
		(adjacent sq6_23 sq5_23)
		(adjacent sq6_23 sq7_23)
		(adjacent sq6_25 sq7_25)
		(adjacent sq6_25 sq6_26)
		(adjacent sq6_27 sq5_27)
		(adjacent sq6_27 sq7_27)
		(adjacent sq6_27 sq6_26)
		(adjacent sq6_26 sq6_25)
		(adjacent sq6_26 sq6_27)
		(adjacent sq11_10 sq10_10)
		(adjacent sq11_10 sq11_11)
		(adjacent sq1_25 sq0_25)
		(adjacent sq1_25 sq2_25)
		(adjacent sq1_27 sq0_27)
		(adjacent sq1_27 sq2_27)
		(adjacent sq11_17 sq10_17)
		(adjacent sq4_16 sq4_15)
		(adjacent sq4_16 sq4_17)
		(adjacent sq4_17 sq3_17)
		(adjacent sq4_17 sq4_16)
		(adjacent sq4_14 sq3_14)
		(adjacent sq4_14 sq4_15)
		(adjacent sq4_15 sq4_14)
		(adjacent sq4_15 sq4_16)
		(adjacent sq4_12 sq4_11)
		(adjacent sq6_19 sq5_19)
		(adjacent sq6_19 sq6_18)
		(adjacent sq4_10 sq3_10)
		(adjacent sq4_10 sq5_10)
		(adjacent sq4_10 sq4_11)
		(adjacent sq4_11 sq4_10)
		(adjacent sq4_11 sq4_12)
		(adjacent sq6_15 sq6_16)
		(adjacent sq6_16 sq6_15)
		(adjacent sq6_16 sq6_17)
		(adjacent sq6_17 sq7_17)
		(adjacent sq6_17 sq6_16)
		(adjacent sq6_17 sq6_18)
		(adjacent sq6_10 sq5_10)
		(adjacent sq6_10 sq6_9)
		(adjacent sq6_10 sq6_11)
		(adjacent sq5_10 sq4_10)
		(adjacent sq5_10 sq6_10)
		(adjacent sq6_12 sq7_12)
		(adjacent sq6_12 sq6_11)
		(adjacent sq4_19 sq5_19)
		(adjacent sq4_19 sq4_20)
		(adjacent sq3_17 sq2_17)
		(adjacent sq3_17 sq4_17)
		(adjacent sq3_14 sq2_14)
		(adjacent sq3_14 sq4_14)
		(adjacent sq3_10 sq2_10)
		(adjacent sq3_10 sq4_10)
		(adjacent sq9_10 sq8_10)
		(adjacent sq9_10 sq10_10)
		(adjacent sq9_15 sq8_15)
		(adjacent sq9_15 sq10_15)
		(adjacent sq9_17 sq8_17)
		(adjacent sq9_17 sq10_17)
		(adjacent sq9_2 sq8_2)
		(adjacent sq9_2 sq10_2)
		(adjacent sq5_23 sq4_23)
		(adjacent sq5_23 sq6_23)
		(adjacent sq5_27 sq4_27)
		(adjacent sq5_27 sq6_27)
	)
	(:goal
		(and
		(empty-at-position sq0_0)
		(empty-at-position sq0_27)
		(empty-at-position sq11_0)
		(empty-at-position sq11_27)
		)
	)
)