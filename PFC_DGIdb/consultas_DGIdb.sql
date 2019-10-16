/* Publicaciones disponibles (datos completos e incompletos): 36979 */
-- create view publicaciones_ifg as
select g.id as gen_id, g.name as gen, d.id as droga_id, d.name as droga, ict.id as interaccion_id, ict.type as interaccion, p.pmid as publicacion
from interaction_claims ic
left join gene_claims gc on ic.gene_claim_id = gc.id
left join drug_claims dc on ic.drug_claim_id = dc.id
left join interaction_claims_publications icp on icp.interaction_claim_id = ic.id
left join publications p on icp.publication_id = p.id
left join interaction_claim_types_interaction_claims ictic on ictic.interaction_claim_id = ic.id
left join interaction_claim_types ict on ictic.interaction_claim_type_id = ict.id
left join genes g on gc.gene_id = g.id
left join drugs d on dc.drug_id = d.id
where p.pmid is not null
group by g.id,gen,d.id,droga,ict.id,interaccion,publicacion 
order by gen,droga,interaccion,publicacion 

-- EXCEPT

/* Datos completos disponibles (gen,droga,interaccion,pmid): 10175 */
-- create view interaccion_farmaco_gen as
select g.id as gen_id, g.name as gen, d.id as droga_id, d.name as droga, ict.id as interaccion_id, ict.type as interaccion, p.pmid as publicacion
from interaction_claims ic
join gene_claims gc on ic.gene_claim_id = gc.id
join drug_claims dc on ic.drug_claim_id = dc.id
join interaction_claims_publications icp on icp.interaction_claim_id = ic.id
join publications p on icp.publication_id = p.id
join interaction_claim_types_interaction_claims ictic on ictic.interaction_claim_id = ic.id
join interaction_claim_types ict on ictic.interaction_claim_type_id = ict.id
join genes g on gc.gene_id = g.id
join drugs d on dc.drug_id = d.id
where p.pmid is not null
group by g.id,gen,d.id,droga,ict.id,interaccion,publicacion 
order by gen,droga,interaccion,publicacion 

-- select * from interaccion_farmaco_gen

-- select * from publicaciones_ifg except select * from interaccion_farmaco_gen

/* Número total de genes registrados en DGIdb: 41102 */
select g.name as gen from genes g
group by gen
order by gen

/* Número de genes en fuentes disponibles: 899 */
-- create view genes_ifg as
select gen_id,gen from interaccion_farmaco_gen
group by gen_id,gen
order by gen,gen_id

/* Número total de drogas registradas en DGIdb: 9501 */
select d.name as droga from drugs d
group by droga
order by droga

/* Número de drogas en fuentes disponibles: 1119 */
-- create view drogas_ifg as
select droga_id,droga from interaccion_farmaco_gen
group by droga_id,droga
order by droga,droga_ida

/* Número total de interacciones registradas en DGIdb: 51 */
select ict.type as interaccion from interaction_claim_types ict
group by interaccion
order by interaccion

/* Número de interacciones en fuentes disponibles: 34 */
-- create view interacciones_ifg as
select interaccion_id,interaccion from interaccion_farmaco_gen
group by interaccion_id,interaccion
order by interaccion,interaccion_id

/* Aliases de las drogas disponibles: 149169 */
-- create view drogas_alias_ifg as
select difg.droga, dca.alias
from drug_claim_aliases dca
join drug_claims dc on dca.drug_claim_id = dc.id
join drogas_ifg difg on dc.drug_id = difg.droga_id
group by difg.droga, dca.alias
order by difg.droga, dca.alias

/* Aliases de las genes disponibles: 12057 */
-- create view genes_alias_ifg as
select gifg.gen, gca.alias
from gene_claim_aliases gca
join gene_claims gc on gca.gene_claim_id = gc.id
join genes_ifg gifg on gc.gene_id = gifg.gen_id
group by gifg.gen, gca.alias
order by gifg.gen, gca.alias