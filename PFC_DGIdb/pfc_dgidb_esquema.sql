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

-- copy (
-- select g.nombre as gen, ga.alias as alias
-- from genes g
-- join gen_alias ga on ga.id_gen = g.id
-- group by gen,alias
-- order by gen,alias
-- ) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\gen_alias.csv' csv header

-- copy (
-- select g.nombre as gen, ga.alias as alias
-- from genes g
-- join gen_alias ga on ga.id_gen = g.id
-- group by gen,alias
-- order by gen,alias
-- ) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\gen_alias.tsv' WITH NULL AS ''

-- copy (
-- select d.nombre as droga, da.alias as alias
-- from drogas d
-- join droga_alias da on da.id_droga = d.id
-- group by droga,alias
-- order by droga,alias
-- ) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\droga_alias.csv' csv header

-- copy (
-- select d.nombre as droga, da.alias as alias
-- from drogas d
-- join droga_alias da on da.id_droga = d.id
-- group by droga,alias
-- order by droga,alias
-- ) to 'D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\droga_alias.tsv' WITH NULL AS ''

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

-- Publicaciones que no tienen abstract: 89
UPDATE publicaciones SET abstract = false WHERE pmid IN (11134, 29252, 458448, 570044, 689791, 1207670, 1350385, 1654268, 1678274, 1731757, 1852778, 1986546, 2157221, 2611529, 3060178, 3770229, 4152054, 4156788, 4406367, 4539395, 4555460, 4719131, 4861216, 4887393, 4889058, 5284360, 5692397, 5833399, 6010427, 6131674, 6245760, 6336597, 6351026, 6364362, 6897316, 7725982, 7873420, 8153059, 8576907, 8777582, 8813989, 8878254, 9038626, 9341357, 9666280, 10386066, 10715145, 10794682, 11311067, 11378004, 11740746, 12083499, 12538756, 12591363, 12815153, 13438725, 13505370, 14035428, 14752016, 14993472, 15674127, 15679624, 15795321, 16480143, 16547811, 16702197, 16973761, 17202450, 17351742, 17823646, 17824139, 17881754, 18780321, 19075491, 19098484, 19293512, 19458366, 19461861, 19465361, 19604717, 20187262, 20345211, 20378572, 20399934, 20525913, 20562554, 21719882, 22917017, 23289116);

-- Cantidad de publicaciones con las etiquetas completas: 5115 (menos 89 que no tiene abstract y tampoco publicación). 5115-89 = 5026
select p.pmid as pmid
from interacciones_farmaco_gen ifg
join genes g on g.id = ifg.id_gen
join drogas d on d.id = ifg.id_droga
join interacciones i on i.id = ifg.id_interaccion
join publicaciones p on p.id = ifg.id_publicacion
group by pmid
order by pmid

-- Cantidad de interacciones fármaco-gen etiquetadas: 10175
select p.pmid as pmid,g.nombre as gen,d.nombre as droga,i.nombre as interaccion
from interacciones_farmaco_gen ifg
join genes g on g.id = ifg.id_gen
join drogas d on d.id = ifg.id_droga
join interacciones i on i.id = ifg.id_interaccion
join publicaciones p on p.id = ifg.id_publicacion
group by pmid,gen,droga,interaccion
order by pmid,gen,droga,interaccion

-- Cantidad de interacciones fámaco-gen etiquetadas que cuentan con abstrat: 10050
-- copy (
-- select  p.pmid as pmid,g.nombre as gen,d.nombre as droga,i.nombre as interaccion
-- from interacciones_farmaco_gen ifg
-- join genes g on g.id = ifg.id_gen
-- join drogas d on d.id = ifg.id_droga
-- join interacciones i on i.id = ifg.id_interaccion
-- join publicaciones p on p.id = ifg.id_publicacion
-- where p.abstract = true
-- group by pmid,gen,droga,interaccion
-- order by pmid,gen,droga,interaccion
-- ) to 'D:\Descargas\Python\PFC_DGIdb_src\ifg.csv' csv header

update interacciones_farmaco_gen set id_interaccion = 14 where id_interaccion = 15

delete from interacciones where id = 15