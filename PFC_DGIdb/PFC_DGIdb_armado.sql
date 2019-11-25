-- Carga de datos en PFC_DGIdb:

-- Se cargan los genes: tabla gen -> Import -> dgidb_export_genes.csv -> nombre

\copy gen (nombre) from 'dgidb_export_genes.csv' csv;


-- Se cargan las drogas: tabla droga -> Import -> dgidb_export_drogas.csv -> nombre

\copy droga (nombre) from 'dgidb_export_drogas.csv' csv;


-- Se cargan las publicaciones: tabla publicacion -> Import -> dgidb_export_publicaciones.csv -> pmid

\copy publicacion (pmid) from 'dgidb_export_publicaciones.csv' csv;


-- Se cargan las interacciones: tabla interaccion -> Import -> dgidb_export_interacciones.csv -> nombre

\copy interaccion (nombre) from 'dgidb_export_interacciones.csv' csv;


-- Para cargar los alias de droga se crea la tabla auxiliar droga_alias_aux:

create table droga_alias_aux (
	id serial primary key,
	droga text not null,
	alias text not null
);


-- Se cargan los alias de droga: tabla droga_alias_aux -> Import -> formato_insercion_alias_droga.csv -> droga,alias

\copy droga_alias_aux (droga, alias) from 'formato_insercion_alias_droga.csv' csv;


-- Se utilizan los datos de droga_alias_aux para llenar droga_alias:

insert into droga_alias(alias,id_droga)
select daa.alias,d.id
from droga d
join droga_alias_aux daa on d.nombre = daa.droga;


-- Se elimina la tabla auxiliar:
drop table droga_alias_aux;


-- Para cargar los alias de gen se crea la tabla auxiliar gen_alias_aux:

create table gen_alias_aux (
	id serial primary key,
	gen text not null,
	alias text not null
);


-- Se cargan los alias de gen: tabla gen_alias_aux -> Import -> formato_insercion_alias_gen.csv -> gen,alias
\copy gen_alias_aux (gen, alias) from 'formato_insercion_alias_gen.csv' csv;


-- Se utilizan los datos de gen_alias_aux para llenar gen_alias:

insert into gen_alias(alias,id_gen)
select gaa.alias,g.id
from gen g
join gen_alias_aux gaa on g.nombre = gaa.gen;


-- Se elimina la tabla auxiliar:
drop table gen_alias_aux;


-- Para cargar la información de las interaciones fármaco-gen se crea la tabla auxiliar ifg_aux:

create table ifg_aux (
	id serial primary key,
	gen text not null,
	droga text not null,
	interaccion text null,
	publicacion int not null
);


-- Se cargan las interacciones fármaco-gen: tabla ifg_aux -> Import -> dgidb_export_ifg.csv -> gen,droga,interaccion,publicacion

\copy ifg_aux (gen, droga, interaccion, publicacion) from 'dgidb_export_ifg.csv' csv;


-- Se utilizan los datos de ifg_aux para llenar interaccion_farmaco_gen:

insert into interaccion_farmaco_gen(id_gen,id_droga,id_interaccion,id_publicacion)
select g.id,d.id,i.id,p.id
from ifg_aux ifga
join gen g on ifga.gen = g.nombre
join droga d on ifga.droga = d.nombre
join interaccion i on ifga.interaccion = i.nombre
join publicacion p on ifga.publicacion = p.pmid;


-- Se elimina la tabla auxiliar:
drop table ifg_aux;


-- Se unifican las interacciones 'Inhibitor' e 'inhibitor':

update interaccion_farmaco_gen set id_interaccion = 14 where id_interaccion = 15;
delete from interaccion where id = 15;
