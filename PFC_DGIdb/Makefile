PSQL=psql -U postgres -h localhost
PSQL_PFC=${PSQL} -d PFC_DGIdb

help:
	@echo 'Ayuda:'
	@echo 'make dgidb: crea y llena la base de datos DGIdb.'
	@echo 'make pfc: crea y llena la base de datos PFC_DGIdb.'

pfc: PFC_DGIdb_schema.sql formato_insercion_alias_droga.csv formato_insercion_alias_gen.csv dgidb_export_ifg.csv
	# crear la base PFC_DGIdb primero
	# ${PSQL_PFC} -f $<
	# ${PSQL_PFC} -c "\copy droga_alias_aux (droga, alias) from 'formato_insercion_alias_droga.csv' csv"
	# ${PSQL_PFC} -c "\copy gen_alias_aux (gen, alias) from 'formato_insercion_alias_gen.csv' csv"
	${PSQL_PFC} -c "\copy ifg_aux (gen, droga, interaccion, publicacion) from 'dgidb_export_ifg.csv' csv"

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

.PHONY: help dgidb pfc