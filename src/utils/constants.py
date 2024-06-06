from rdflib import URIRef

ONTOLOGY_DETAILS = {"filepath": "./../data/ontology.ttl",
                    "namespace": "http://cltl-hs.org/",
                    "prefix": "hs"}
GAF_DENOTEDIN = URIRef("http://groundedannotationframework.org/gaf#denotedIn")
GAF_DENOTEDBY = URIRef("http://groundedannotationframework.org/gaf#denotedBy")
GAF_CONTAINSDEN = URIRef("http://groundedannotationframework.org/gaf#containsDenotation")
GRASP_ATTFOR = URIRef("http://groundedannotationframework.org/grasp#isAttributionFor")
GRASP_ATTTO = URIRef("http://groundedannotationframework.org/grasp#wasAttributedTo")
SEM_TST = URIRef("http://semanticweb.cs.vu.nl/2009/11/sem/hasBeginTimeStamp")
HS_ID = URIRef("http://cltl-hs.org/id")
INSTANCE_GRAPH = URIRef("http://cltl.nl/leolani/world/Instances")
PERSPECTIVE_GRAPH = URIRef("http://cltl.nl/leolani/talk/Perspectives")
ONTOLOGY_GRAPH = URIRef("http://cltl.nl/leolani/world/Ontology")
