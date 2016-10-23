(define (domain pacman)

	(:requirements 
		:typing
	)

	(:types
		position
	)

	(:predicates
		(pacman-at-square ?x - position) (food-at-square ?x - position) (adjacent ?x ?y - position) 	
	) 

	(:action move
		:parameters (?current-position ?next-position - position)
		:precondition ( and (pacman-at-square ?current-position)
						(adjacent ?current-position ?next-position)
						)
		:effect (and (pacman-at-square ?next-position)
				(not (pacman-at-square ?current-position))
				(not (food-at-square ?next-position))
				)
	)
)

