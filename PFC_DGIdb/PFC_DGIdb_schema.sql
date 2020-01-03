-- esquema para la base PFC_DGIdb:

create table gen (
	id serial primary key,
	nombre varchar(60) not null
);

create table droga (
	id serial primary key,
	nombre text not null
);

create table interaccion (
	id serial primary key,
	nombre varchar(60) not null
);

create table publicacion (
	id serial primary key,
	pmid int not null,
	abstract text not null default true
);

create table gen_alias (
	id serial primary key,
	alias text not null,
	id_gen int not null,
	
	constraint fk1_gen_alias_gen foreign key (id_gen) references gen(id)
);

create table droga_alias (
	id serial primary key,
	alias text not null,
	id_droga int not null,
	
	constraint fk1_droga_alias_droga foreign key (id_droga) references droga(id)
);

create table interaccion_farmaco_gen (
	id serial primary key,
	id_gen int not null,
	id_droga int not null,
	id_interaccion int not null,
	id_publicacion int not null,
	
	constraint fk1_interacciones_farmaco_gen_gen foreign key (id_gen) references gen(id),
	constraint fk2_interacciones_farmaco_gen_droga foreign key (id_droga) references droga(id),
	constraint fk3_interacciones_farmaco_gen_interaccion foreign key (id_interaccion) references interaccion(id),
	constraint fk4_interacciones_farmaco_gen_publicacion foreign key (id_publicacion) references publicacion(id)
);
