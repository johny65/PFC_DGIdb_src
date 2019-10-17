/* Tablas */

create table genes (
	id serial primary key,
	nombre varchar(60) not null
);

create table drogas (
	id serial primary key,
	nombre varchar(60) not null
);

create table interacciones (
	id serial primary key,
	nombre varchar(60) not null
);

create table publicaciones (
	id serial primary key,
	pmid int not null,
	nombre_url varchar(60) not null
);

create table gen_alias (
	id serial primary key,
	alias text not null,
	id_gen int not null,
	
	constraint fk1_gen_alias_genes foreign key (id_gen) references genes(id)
);

create table droga_alias (
	id serial primary key,
	alias text not null,
	id_droga int not null,
	
	constraint fk1_droga_alias_drogas foreign key (id_droga) references drogas(id)
);

create table interacciones_farmaco_gen (
	id serial primary key,
	id_gen int not null,
	id_droga int not null,
	id_interaccion int not null,
	id_publicacion int not null,
	
	constraint fk1_interacciones_farmaco_gen_genes foreign key (id_gen) references genes(id),
	constraint fk2_interacciones_farmaco_gen_droga foreign key (id_droga) references drogas(id),
	constraint fk3_interacciones_farmaco_gen_interacciones foreign key (id_interaccion) references interacciones(id),
	constraint fk4_interacciones_farmaco_gen_publicaciones foreign key (id_publicacion) references publicaciones(id)
);

/* Datos */

-- copy (select gen from genes_ifg) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\genes.csv' csv header
-- copy (select droga from drogas_ifg) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\drogas.csv' csv header
-- copy (select interaccion from interacciones_ifg) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\interacciones.csv' csv header
-- copy (select droga,alias from drogas_alias_ifg) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\drogas_alias.csv' csv header
-- copy (select gen,alias from genes_alias_ifg) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\genes_alias.csv' csv header
-- copy (select gen,droga,interaccion,publicacion from interaccion_farmaco_gen) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\interaccion_farmaco_gen.csv' csv header

 --

create table droga_alias_aux (
	id serial primary key,
	droga varchar(60) not null,
	alias text not null
);

insert into droga_alias(alias,id_droga)
select daa.alias,d.id
from drogas d
join droga_alias_aux daa on d.nombre = daa.droga

drop table droga_alias_aux

--

create table gen_alias_aux (
	id serial primary key,
	gen varchar(60) not null,
	alias text not null
);

insert into gen_alias(alias,id_gen)
select gaa.alias,g.id
from genes g
join gen_alias_aux gaa on g.nombre = gaa.gen

drop table gen_alias_aux

--

create table ifg_aux (
	id serial primary key,
	gen varchar(60) not null,
	droga varchar(60) not null,
	interaccion varchar(60) not null,
	publicacion int not null
);

insert into interacciones_farmaco_gen(id_gen,id_droga,id_interaccion,id_publicacion)
select g.id,d.id,i.id,p.id
from ifg_aux ifga
join genes g on ifga.gen = g.nombre
join drogas d on ifga.droga = d.nombre
join interacciones i on ifga.interaccion = i.nombre
join publicaciones p on ifga.publicacion = p.pmid

drop table ifg_aux

--

