(define (problem pacman-eat-all)
	(:domain pacman)
	(:objects
		 sq4_4 sq4_5 sq4_6 sq2_14 sq4_0 sq4_1 sq4_2 sq4_3 sq4_8 sq4_9 sq4_7 sq4_12 sq4_13 sq0_0 sq0_1 sq0_2 sq0_3 sq0_4 sq0_5 sq0_6 sq0_7 sq0_8 sq0_9 sq4_17 sq4_14 sq4_16 sq1_9 sq2_7 sq2_4 sq2_5 sq2_2 sq2_3 sq2_0 sq2_1 sq1_0 sq1_3 sq2_8 sq1_6 sq2_10 sq1_12 sq1_15 sq4_10 sq1_17 sq2_11 sq4_11 sq2_12 sq3_17 sq2_13 sq2_15 sq0_12 sq0_13 sq0_10 sq0_11 sq0_16 sq0_17 sq0_15 sq2_17 sq2_6 sq2_9 sq4_15 - position
	)
	(:init
		(pacman-at-position sq2_8)
		(empty-at-position sq4_5)
		(empty-at-position sq4_6)
		(empty-at-position sq2_14)
		(empty-at-position sq4_8)
		(empty-at-position sq4_9)
		(empty-at-position sq4_7)
		(empty-at-position sq4_12)
		(empty-at-position sq4_13)
		(empty-at-position sq0_1)
		(empty-at-position sq0_2)
		(empty-at-position sq0_3)
		(empty-at-position sq0_4)
		(empty-at-position sq0_5)
		(empty-at-position sq0_6)
		(empty-at-position sq0_7)
		(empty-at-position sq0_8)
		(empty-at-position sq0_9)
		(empty-at-position sq4_17)
		(empty-at-position sq4_14)
		(empty-at-position sq4_16)
		(empty-at-position sq2_7)
		(empty-at-position sq2_4)
		(empty-at-position sq2_5)
		(empty-at-position sq2_2)
		(empty-at-position sq2_3)
		(empty-at-position sq2_0)
		(empty-at-position sq2_1)
		(empty-at-position sq2_8)
		(empty-at-position sq2_10)
		(empty-at-position sq1_15)
		(empty-at-position sq4_10)
		(empty-at-position sq1_17)
		(empty-at-position sq2_11)
		(empty-at-position sq4_11)
		(empty-at-position sq2_12)
		(empty-at-position sq3_17)
		(empty-at-position sq2_13)
		(empty-at-position sq2_15)
		(empty-at-position sq0_10)
		(empty-at-position sq0_11)
		(empty-at-position sq0_16)
		(empty-at-position sq0_17)
		(empty-at-position sq0_15)
		(empty-at-position sq2_17)
		(empty-at-position sq2_6)
		(empty-at-position sq2_9)
		(empty-at-position sq4_15)
		(adjacent sq4_4 sq4_3)
		(adjacent sq4_4 sq4_5)
		(adjacent sq4_5 sq4_4)
		(adjacent sq4_5 sq4_6)
		(adjacent sq4_6 sq4_5)
		(adjacent sq4_6 sq4_7)
		(adjacent sq2_14 sq2_13)
		(adjacent sq2_14 sq2_15)
		(adjacent sq4_0 sq4_1)
		(adjacent sq4_1 sq4_0)
		(adjacent sq4_1 sq4_2)
		(adjacent sq4_2 sq4_1)
		(adjacent sq4_2 sq4_3)
		(adjacent sq4_3 sq4_2)
		(adjacent sq4_3 sq4_4)
		(adjacent sq4_8 sq4_7)
		(adjacent sq4_8 sq4_9)
		(adjacent sq4_9 sq4_8)
		(adjacent sq4_9 sq4_10)
		(adjacent sq4_7 sq4_6)
		(adjacent sq4_7 sq4_8)
		(adjacent sq4_12 sq4_11)
		(adjacent sq4_12 sq4_13)
		(adjacent sq4_13 sq4_12)
		(adjacent sq4_13 sq4_14)
		(adjacent sq0_0 sq1_0)
		(adjacent sq0_0 sq0_1)
		(adjacent sq0_1 sq0_0)
		(adjacent sq0_1 sq0_2)
		(adjacent sq0_2 sq0_1)
		(adjacent sq0_2 sq0_3)
		(adjacent sq0_3 sq1_3)
		(adjacent sq0_3 sq0_2)
		(adjacent sq0_3 sq0_4)
		(adjacent sq0_4 sq0_3)
		(adjacent sq0_4 sq0_5)
		(adjacent sq0_5 sq0_4)
		(adjacent sq0_5 sq0_6)
		(adjacent sq0_6 sq1_6)
		(adjacent sq0_6 sq0_5)
		(adjacent sq0_6 sq0_7)
		(adjacent sq0_7 sq0_6)
		(adjacent sq0_7 sq0_8)
		(adjacent sq0_8 sq0_7)
		(adjacent sq0_8 sq0_9)
		(adjacent sq0_9 sq1_9)
		(adjacent sq0_9 sq0_8)
		(adjacent sq0_9 sq0_10)
		(adjacent sq4_17 sq3_17)
		(adjacent sq4_17 sq4_16)
		(adjacent sq4_14 sq4_13)
		(adjacent sq4_14 sq4_15)
		(adjacent sq4_16 sq4_15)
		(adjacent sq4_16 sq4_17)
		(adjacent sq1_9 sq0_9)
		(adjacent sq1_9 sq2_9)
		(adjacent sq2_7 sq2_6)
		(adjacent sq2_4 sq2_3)
		(adjacent sq2_4 sq2_5)
		(adjacent sq2_5 sq2_4)
		(adjacent sq2_5 sq2_6)
		(adjacent sq2_2 sq2_1)
		(adjacent sq2_2 sq2_3)
		(adjacent sq2_3 sq1_3)
		(adjacent sq2_3 sq2_2)
		(adjacent sq2_3 sq2_4)
		(adjacent sq2_0 sq1_0)
		(adjacent sq2_0 sq2_1)
		(adjacent sq2_1 sq2_0)
		(adjacent sq2_1 sq2_2)
		(adjacent sq1_0 sq0_0)
		(adjacent sq1_0 sq2_0)
		(adjacent sq1_3 sq0_3)
		(adjacent sq1_3 sq2_3)
		(adjacent sq2_8 sq2_7)
		(adjacent sq2_8 sq2_9)
		(adjacent sq1_6 sq0_6)
		(adjacent sq1_6 sq2_6)
		(adjacent sq2_10 sq2_9)
		(adjacent sq2_10 sq2_11)
		(adjacent sq1_12 sq0_12)
		(adjacent sq1_12 sq2_12)
		(adjacent sq1_15 sq0_15)
		(adjacent sq1_15 sq2_15)
		(adjacent sq4_10 sq4_9)
		(adjacent sq4_10 sq4_11)
		(adjacent sq1_17 sq0_17)
		(adjacent sq1_17 sq2_17)
		(adjacent sq2_11 sq2_10)
		(adjacent sq2_11 sq2_12)
		(adjacent sq4_11 sq4_10)
		(adjacent sq4_11 sq4_12)
		(adjacent sq2_12 sq1_12)
		(adjacent sq2_12 sq2_11)
		(adjacent sq2_12 sq2_13)
		(adjacent sq3_17 sq2_17)
		(adjacent sq3_17 sq4_17)
		(adjacent sq2_13 sq2_12)
		(adjacent sq2_13 sq2_14)
		(adjacent sq2_15 sq1_15)
		(adjacent sq2_15 sq2_14)
		(adjacent sq0_12 sq1_12)
		(adjacent sq0_12 sq0_11)
		(adjacent sq0_12 sq0_13)
		(adjacent sq0_13 sq0_12)
		(adjacent sq0_10 sq0_9)
		(adjacent sq0_10 sq0_11)
		(adjacent sq0_11 sq0_10)
		(adjacent sq0_11 sq0_12)
		(adjacent sq0_16 sq0_15)
		(adjacent sq0_16 sq0_17)
		(adjacent sq0_17 sq1_17)
		(adjacent sq0_17 sq0_16)
		(adjacent sq0_15 sq1_15)
		(adjacent sq0_15 sq0_16)
		(adjacent sq2_17 sq1_17)
		(adjacent sq2_17 sq3_17)
		(adjacent sq2_6 sq1_6)
		(adjacent sq2_6 sq2_5)
		(adjacent sq2_6 sq2_7)
		(adjacent sq2_9 sq1_9)
		(adjacent sq2_9 sq2_10)
		(adjacent sq4_15 sq4_14)
		(adjacent sq4_15 sq4_16)
	)
	(:goal
		(and
		(empty-at-position sq0_0)
		(empty-at-position sq0_12)
		(empty-at-position sq0_13)
		(empty-at-position sq1_0)
		(empty-at-position sq1_3)
		(empty-at-position sq1_6)
		(empty-at-position sq1_9)
		(empty-at-position sq1_12)
		(empty-at-position sq4_0)
		(empty-at-position sq4_1)
		(empty-at-position sq4_2)
		(empty-at-position sq4_3)
		(empty-at-position sq4_4)
		)
	)
)