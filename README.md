# PFC_DGIdb_src

Fuentes para el proyecto final.


## Resumen:

Etiquetas:                          Papers:
Casos de ejemplos: 10175            PDF scrapeados:     4021
Publicaciones:      5115            Como TXT:           3825
Genes:               899
Drogas:             1119            Totales:
Tipos interacción:    33            Genes:              2440
                                    Drogas:             3568
Entrenamiento:
Ejemplos finales:   3144 (1)
Sintéticos:        23106 (2)
Ejemplos totales:  26250

Notas:
(1): son los casos de ejemplo que quedan de los 10175 después de procesar el artículo, es decir,
     casos que sirven ya que el gen/droga de interés sí se menciona en el texto.
(2): son los casos de ejemplo generados de tipo "sin_interacción", explicado más abajo.


## Detalles:

### Publicaciones

A partir de la base de datos DGIdb se obtuvieron 10175 casos etiquetados de interacciones fármaco-gen (cantidad de filas en la tabla `interaccion_farmaco_gen`, con información de PMID, gen, droga e interacción).

Todos estos casos surgen de un conjunto de 5115 publicaciones distintas. Estos 5115 PMID están registrados en el archivo `PFC_DGIdb/pmids_etiquetas_completas.csv` y se obtienen con la siguiente consulta:

    select distinct p.pmid
    from interaccion_farmaco_gen ifg
    join publicacion p on ifg.id_publicacion = p.id
    order by pmid

De la misma manera se obtiene que hay 899 genes distintos en juego, 1119 drogas y 33 tipos de interacción.

## Interacciones

Con la siguiente consulta se obtiene cuántos casos de ejemplo hay por interacción:

    select i.nombre, count(ifg.id)
    from interaccion_farmaco_gen ifg
    join interaccion i on ifg.id_interaccion = i.id
    group by i.nombre
    order by 2 desc

Esta información fue exportada en el archivo `PFC_DGIdb/ejemplos_x_interacciones.csv`.

### Scraping

De estas publicaciones se pudieron scrapear 4021 documentos PDF. Convertidos a TXT quedó un total de 3825 archivos, a los cuales se les hizo un proceso de "ungreek" (reemplazar las apariciones de letras griegas por sus nombres).

Además se scrapearon abstracts, títulos y palabras clave de las publicaciones (archivo `scraping/pmids_titulos_abstracts_keywords.csv`). Este archivo también fue "ungreek-ed" y se le limpió el HTML.

Luego de esta etapa todo el contenido quedó en minúsculas, sin letras griegas y sin etiquetas HTML que pudieran perjudicar la búsqueda.

### Búsqueda y reemplazo

El objetivo de esta etapa es reemplazar todas las apariciones de genes y drogas en los textos, teniendo en cuenta los distintos alias que tiene cada uno, por una identificación unívoca de cada uno. De esta forma cada gen y droga es luego fácilmente identificable.

Para esto se usa el ID del gen/droga. Se guarda el mapeo del ID al nombre en los archivos `PFC_DGIdb/ids_gen.csv` y `PFC_DGIdb/ids_droga.csv`, obtenidos con la siguiente consulta:

    copy (
        select concat('g', id), lower(nombre) from gen order by nombre
    ) to 'ids_gen.csv' csv;

    copy (
        select concat('d', id), lower(nombre) from droga order by nombre
    ) to 'ids_droga.csv' csv;

Aparte se scrapearon alias de genes de GeneCards.com, y con toda esta información se generaron los archivos `PFC_DGIdb/alias_gen.csv` y `PFC_DGIdb/alias_droga.csv` (ambos limpiados: minúsculas, ungreek, sin HTML).

Las etiquetas de cada ejemplo se exportaron al archivo `PFC_DGIdb/pfc_dgidb_export_ifg.csv` con la siguiente consulta:

    copy (
        select p.pmid, concat('g', g.id), concat('d', d.id), lower(i.nombre)
        from interaccion_farmaco_gen ifg
        join publicacion p on p.id = ifg.id_publicacion
        join gen g on g.id = ifg.id_gen
        join droga d on d.id = ifg.id_droga
        join interaccion i on i.id = ifg.id_interaccion
        order by 1, 2, 3, 4
    ) to '/tmp/pfc_dgidb_export_ifg.csv' csv

#### Búsqueda de ocurrencias:

Luego se realizó la búsqueda de cada gen y droga en los textos de 3 maneras distintas: dejando de lado alias que tienen embeddings (pueden ser palabras comunes) y dejando de lado alias repetidos (alias que corresponden a más de un gen o droga a la vez); dejando de lado con embedding e incluyendo con repeticiones; y con todo incluido. Estas búsquedas se corresponden a los archivos `entidad_se_sr.csv`, `entidad_se_cr.csv` y `entidad_ce_cr.csv`.

#### Reemplazo:

Para el reemplazo en sí, se reemplazó la aparición de cada alias de gen/droga por una cadena de la forma "xxxidentificadorxxx", primero teniendo en cuenta sólo las ocurrencias SE/SR; si las etiquetas no se encontraron en el texto con estas ocurrencias, entonces se intentó con las SE/CR; si igualmente no se encontraron, entonces se prosiguió con todas.

Luego del proceso de reemplazo se obtuvieron 5115 archivos de texto (los artículos que no poseían documento quedaron con su título/abstract/palabras clave).

### Generación de ejemplos negativos



fdssfdsg
sdgfdgfdgfd
fgfhgh



26510944,g1108,d443,agonist
26510944,g1108,d443,partial agonist
26510944,g1113,d443,antagonist
26510944,g1114,d443,antagonis

