PSQL=psql -U postgres -h localhost

help:
	@echo 'Ayuda:'
	@echo 'make dgidb: crea y llena la base de datos DGIdb.'

dgidb: data.sql structure.sql
	# crear la base primero
	${PSQL} -d dgidb -f structure.sql
	${PSQL} -d dgidb -f data.sql

data.sql:
	# descargar de algún lado
	echo "nop"

structure.sql:
	# descargar de algún lado
	echo "nop"

.PHONY: help dgidb