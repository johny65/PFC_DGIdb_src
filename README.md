# PFC_DGIdb_src

Fuentes para el proyecto final.


## Resumen:

Etiquetas:                          Papers:
Casos de ejemplos: 10175            PDF scrapeados: 4021
Publicaciones:      5115            Como TXT:       3897
Genes:               899
Drogas:             1119
Tipos interacción:    33

Totales:
Genes:              2440
Drogas:             3568


## Detalles:

### Publicaciones

A partir de la base de datos DGIdb se obtuvieron 10175 casos etiquetados de interacciones fármaco-gen (cantidad de filas en la tabla `interaccion_farmaco_gen`, con información de PMID, gen, droga e interacción).

Todos estos casos surgen de un conjunto de 5115 publicaciones distintas. Estos 5115 PMID están registrados en el archivo `PFC_DGIdb/pmids_etiquetas_completas.csv` y se obtienen con la siguiente consulta:

    select distinct p.pmid
    from interaccion_farmaco_gen ifg
    join publicacion p on ifg.id_publicacion = p.id
    order by pmid

De la misma manera se obtiene que hay 899 genes distintos en juego, 1119 drogas y 33 tipos de interacción.

### Scraping

De estas publicaciones se pudieron scrapear 4021 documentos PDF. Convertidos a TXT quedó un total de 3897 archivos, a los cuales se les hizo un proceso de "ungreek" (reemplazar las apariciones de letras griegas por sus nombres).

Además se scrapearon abstracts, títulos y palabras clave de las publicaciones (archivo `scraping/pmids_titulos_abstracts_keywords.csv`). Este archivo también fue "ungreek-ed" y se le limpió el HTML.

Luego de esta etapa todo el contenido quedó en minúsculas, sin letras griegas y sin etiquetas HTML que pudieran perjudicar la búsqueda.

### Reemplazo

El objetivo de esta etapa es reemplazar todas las apariciones de genes y drogas en los textos, teniendo en cuenta los distintos alias que tiene cada uno, por una identificación unívoca de cada uno. De esta forma cada gen y droga es luego fácilmente identificable.

Para esto se usa el ID del gen/droga. Se guarda el mapeo del ID al nombre en los archivos `PFC_DGIdb/ids_gen.csv` y `PFC_DGIdb/ids_droga.csv`, obtenidos con la siguiente consulta:

    copy (
        select concat('g', id), lower(nombre) from gen order by nombre
    ) to 'ids_gen.csv' csv;

    copy (
        select concat('d', id), lower(nombre) from droga order by nombre
    ) to 'ids_droga.csv' csv;

Aparte se scrapearon alias de genes de GeneCards.com, y con toda esta información se generaron los archivos `PFC_DGIdb/alias_gen.csv` y `PFC_DGIdb/alias_droga.csv` (ambos limpiados: minúsculas, ungreek, sin HTML).

