# se deja el nombre en minúscula porque el comando para crearla no toma en mayúscula:
DB=pfc_dgidb
PSQL=psql -U postgres -h localhost
PSQL_PFC=${PSQL} -d ${DB}

help:
	@echo 'Ayuda:'
	@echo '    make dgidb: crea y llena la base de datos DGIdb.'
	@echo '    make pfcdb: crea y llena la base de datos PFC_DGIdb.'
	@echo '    make clean: limpia la carpeta de temporales.'

pfcdb: PFC_DGIdb_schema.sql PFC_DGIdb_armado.sql
	${PSQL} -c "CREATE DATABASE ${DB}"
	${PSQL_PFC} -f PFC_DGIdb_schema.sql
	${PSQL_PFC} -f PFC_DGIdb_armado.sql

dgidb: tmp_dgidb_repo tmp_dgidb_data_repo
	${PSQL} -c "CREATE DATABASE dgidb"
	${PSQL} -d dgidb -f tmp_dgidb_repo/db/structure.sql
	${PSQL} -d dgidb -f tmp_dgidb_data_repo/data.sql

tmp_dgidb_repo:
	git clone --depth=1 https://github.com/griffithlab/dgi-db $@

tmp_dgidb_data_repo:
	git clone --depth=1 git://git.genomedata.org/dgidb_data.git $@

clean:
	rm -rf tmp_dgidb_repo tmp_dgidb_data_repo

.PHONY: help dgidb pfcdb clean