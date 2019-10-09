select count(*) from publications where pmid is null

select * from sources order by full_name  

select /*ga.alias as gen,*/ g.name, g.long_name, d.name as droga, ict.type, p.pmid from interaction_claims ic
join gene_claims gc on ic.gene_claim_id = gc.id
join drug_claims dc on ic.drug_claim_id = dc.id
join interaction_claims_publications icp on icp.interaction_claim_id = ic.id
join publications p on icp.publication_id = p.id
join interaction_claim_types_interaction_claims ictic on ictic.interaction_claim_id = ic.id
join interaction_claim_types ict on ictic.interaction_claim_type_id = ict.id
join genes g on gc.gene_id = g.id
--join gene_aliases ga on ga.gene_id = g.id
join drugs d on dc.drug_id = d.id

where g.name = 'EGFR' and d.name = 'CETUXIMAB'
limit 1000

/*---*/

/* Número total de interacciones registradas en DGIdb: 51 */
select ict.type as interaccion from interaction_claim_types ict
group by interaccion
order by interaccion

/* Número total de drogas registradas en DGIdb: 9501 */
select d.name as droga from drugs d
group by droga
order by droga

/* Número total de genes registrados en DGIdb: 41102 */
select g.name as gen from genes g
group by gen
order by gen

/* Número total de interacciones fármaco-gen etiquetadas (gen,droga,interacción): 17663 */
select g.name as gen, d.name as droga, ict.type as interaccion from interaction_claims ic
join gene_claims gc on ic.gene_claim_id = gc.id
join drug_claims dc on ic.drug_claim_id = dc.id
join interaction_claim_types_interaction_claims ictic on ictic.interaction_claim_id = ic.id
join interaction_claim_types ict on ictic.interaction_claim_type_id = ict.id
join genes g on gc.gene_id = g.id
join drugs d on dc.drug_id = d.id
group by gen,droga,interaccion
order by gen,droga,interaccion

/* Número de fuentes disponibles (gen,droga,interaccion,pmid) de interacciones fármaco-gen: 10175 */
select g.name as gen, d.name as droga, ict.type as interaccion, p.pmid as id_publicacion from interaction_claims ic
join gene_claims gc on ic.gene_claim_id = gc.id
join drug_claims dc on ic.drug_claim_id = dc.id
join interaction_claims_publications icp on icp.interaction_claim_id = ic.id
join publications p on icp.publication_id = p.id
join interaction_claim_types_interaction_claims ictic on ictic.interaction_claim_id = ic.id
join interaction_claim_types ict on ictic.interaction_claim_type_id = ict.id
join genes g on gc.gene_id = g.id
join drugs d on dc.drug_id = d.id
group by gen,droga,interaccion,id_publicacion
order by gen,droga,interaccion,id_publicacion

/* Número de genes en fuentes disponibles: 899 */
select g.name as gen from interaction_claims ic
join gene_claims gc on ic.gene_claim_id = gc.id
join drug_claims dc on ic.drug_claim_id = dc.id
join interaction_claims_publications icp on icp.interaction_claim_id = ic.id
join publications p on icp.publication_id = p.id
join interaction_claim_types_interaction_claims ictic on ictic.interaction_claim_id = ic.id
join interaction_claim_types ict on ictic.interaction_claim_type_id = ict.id
join genes g on gc.gene_id = g.id
join drugs d on dc.drug_id = d.id
group by gen
order by gen

/* Número de drogas en fuentes disponibles: 1119 */
select d.name as droga from interaction_claims ic
join gene_claims gc on ic.gene_claim_id = gc.id
join drug_claims dc on ic.drug_claim_id = dc.id
join interaction_claims_publications icp on icp.interaction_claim_id = ic.id
join publications p on icp.publication_id = p.id
join interaction_claim_types_interaction_claims ictic on ictic.interaction_claim_id = ic.id
join interaction_claim_types ict on ictic.interaction_claim_type_id = ict.id
join genes g on gc.gene_id = g.id
join drugs d on dc.drug_id = d.id
group by droga
order by droga

/* Número de interacciones en fuentes disponibles: 34 */
select ict.type as interaccion from interaction_claims ic
join gene_claims gc on ic.gene_claim_id = gc.id
join drug_claims dc on ic.drug_claim_id = dc.id
join interaction_claims_publications icp on icp.interaction_claim_id = ic.id
join publications p on icp.publication_id = p.id
join interaction_claim_types_interaction_claims ictic on ictic.interaction_claim_id = ic.id
join interaction_claim_types ict on ictic.interaction_claim_type_id = ict.id
join genes g on gc.gene_id = g.id
join drugs d on dc.drug_id = d.id
group by interaccion
order by interaccion

/*
- Resumen
Datos: 10075/17663
Genes: 899/41102
Drogas: 1119/9501
Interacciones: 34/51
*/