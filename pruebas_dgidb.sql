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
