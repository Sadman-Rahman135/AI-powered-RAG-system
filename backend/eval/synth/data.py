"""Seeded generation of synthetic invoice *content*: parties, dates, line items, taxes, arithmetic, intentional errors.

An :class:`InvoiceSpec` is the single source of truth for one document. The renderers print it, the spreadsheet
writers tabulate it and :meth:`InvoiceSpec.record` turns it into the ground-truth label, so a label can only differ
from the printed document if a renderer is wrong. Every field of the spec holds what is *printed* (``None`` = not
printed); ``finalize`` computes the money fields from the inputs and then applies the intentional error, so the
printed numbers of an erroneous document are the labels (see ``eval.labels`` for the conventions).

Everything is driven by ``random.Random`` instances handed in by the caller — the module has no hidden state.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from app.domain.schemas import DocumentType, LineItem
from eval.labels import LabeledRecord

CENT = Decimal("0.01")
HUNDRED = Decimal(100)


def q2(value: Decimal) -> Decimal:
    """Round half up to two decimals."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def plain(value: Decimal) -> Decimal:
    """Strip trailing zeros without switching to exponent notation (``7.50`` -> ``7.5``, ``100`` stays ``100``)."""
    text = format(value.normalize(), "f")
    return Decimal(text)


# ====================================================================================================================
# markets: names, addresses, phone numbers, tax and bank identifiers
# ====================================================================================================================
@dataclass(frozen=True)
class Market:
    key: str
    country: str
    tld: str
    surnames: tuple[str, ...]
    places: tuple[str, ...]
    cities: tuple[tuple[str, str], ...]  # (city, region or postal prefix)
    streets: tuple[str, ...]
    suffixes: tuple[str, ...]
    sons: str  # "& Sons" in the local language
    local_words: tuple[str, ...] = ()
    first_names: tuple[str, ...] = ("Emily", "Daniel", "Sofia", "Marcus", "Hannah", "Oliver", "Priya", "Lucas", "Grace")


# fmt: off
MARKETS: dict[str, Market] = {
    "us": Market(
        "us", "USA", "com",
        ("Anderson", "Baker", "Carter", "Dawson", "Edwards", "Foster", "Garcia", "Hughes", "Ingram", "Jensen", "Keller",
         "Lopez", "Mitchell", "Nguyen", "Olsen", "Patel", "Quinn", "Reyes", "Sullivan", "Thompson", "Underwood",
         "Vargas", "Walker", "Yates", "Zimmerman", "Brennan", "Coleman", "Delgado", "O'Brien", "D'Angelo", "Holt"),
        ("Brightwater", "Northgate", "Maple Ridge", "Cedar Point", "Summit", "Lakeshore", "Ironwood", "Harborview",
         "Prairie", "Redwood", "Silverline", "Bluestone", "Eastport", "Pinecrest", "Riverbend", "Goldcrest", "Oakmont"),
        (("Springfield, IL", "627"), ("Austin, TX", "787"), ("Denver, CO", "802"), ("Portland, OR", "972"),
         ("Columbus, OH", "432"), ("Raleigh, NC", "276"), ("Phoenix, AZ", "850"), ("Madison, WI", "537"),
         ("Boise, ID", "837"), ("Tampa, FL", "336"), ("Pittsburgh, PA", "152"), ("Minneapolis, MN", "554"),
         ("Charlotte, NC", "282"), ("Albany, NY", "122"), ("Nashville, TN", "372")),
        ("Main St", "Oak Ave", "Industrial Pkwy", "Commerce Dr", "Maple St", "Park Ave", "Elm St", "Washington Blvd",
         "Lakeview Rd", "Market St", "Cedar Ln", "5th Ave", "Mill Rd", "Broadway", "Technology Way"),
        ("Inc.", "LLC", "Corp.", "Co.", "Inc", "LLC", "Ltd."),
        "& Sons",
    ),
    "de": Market(
        "de", "Germany", "de",
        ("Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz", "Hoffmann",
         "Köhler", "Bauer", "Richter", "Klein", "Wolf", "Schröder", "Neumann", "Schwarz", "Zimmermann", "Braun",
         "Krüger", "Hartmann", "Lange", "Werner", "Weiß", "Böhm", "Jäger"),
        ("Rheintal", "Nordsee", "Bayern", "Elbe", "Schwarzwald", "Hansa", "Alpen", "Ostsee"),
        (("Berlin", "101"), ("Hamburg", "200"), ("München", "803"), ("Köln", "505"), ("Frankfurt am Main", "603"),
         ("Stuttgart", "701"), ("Düsseldorf", "402"), ("Leipzig", "041"), ("Nürnberg", "904"), ("Bremen", "282"),
         ("Dresden", "010"), ("Hannover", "301")),
        ("Hauptstraße", "Bahnhofstraße", "Industriestraße", "Gartenweg", "Schillerstraße", "Goethestraße",
         "Lindenallee", "Am Markt", "Berliner Allee", "Kirchplatz"),
        ("GmbH", "AG", "GmbH & Co. KG", "KG", "GmbH", "UG (haftungsbeschränkt)"),
        "& Söhne",
        ("Maschinenbau", "Elektrotechnik", "Großhandel", "Verpackungen", "Spedition", "Gebäudereinigung", "Druckerei",
         "Bürobedarf", "Metallbau", "Handels"),
        ("Anna", "Jonas", "Lena", "Felix", "Katharina", "Tobias", "Julia", "Stefan"),
    ),
    "fr": Market(
        "fr", "France", "fr",
        ("Lefèvre", "Dubois", "Moreau", "Laurent", "Girard", "Fournier", "Mercier", "Lambert", "Rousseau", "Garnier",
         "Chevalier", "François", "Blanc", "Guérin", "Boyer", "Faure", "Bertrand", "Roussel", "Marchand", "L'Hôte"),
        ("Provence", "Aquitaine", "Bretagne", "Alsace", "Normandie", "Occitanie", "Rhône"),
        (("Paris", "750"), ("Lyon", "690"), ("Marseille", "130"), ("Toulouse", "310"), ("Nantes", "440"),
         ("Strasbourg", "670"), ("Bordeaux", "330"), ("Lille", "590"), ("Rennes", "350"), ("Grenoble", "380")),
        ("rue de la République", "avenue Victor Hugo", "boulevard Saint-Germain", "rue du Commerce", "place de la Gare",
         "rue Pasteur", "quai des Chartrons", "avenue des Champs"),
        ("SARL", "SAS", "SA", "SASU", "EURL"),
        "& Fils",
        ("Fournitures", "Services", "Transports", "Équipements", "Informatique", "Nettoyage", "Imprimerie"),
        ("Camille", "Hugo", "Léa", "Mathieu", "Chloé", "Nicolas", "Élodie", "Julien"),
    ),
    "nl": Market(
        "nl", "Netherlands", "nl",
        ("de Vries", "Jansen", "van den Berg", "Bakker", "Visser", "Smit", "Meijer", "de Boer", "Mulder", "de Groot",
         "Bos", "Vos", "Peters", "Hendriks", "van Dijk"),
        ("Amstel", "Rijnmond", "Zeeland", "Veluwe", "Maas"),
        (("Amsterdam", "1012"), ("Rotterdam", "3011"), ("Utrecht", "3511"), ("Eindhoven", "5611"),
         ("Groningen", "9711"), ("Den Haag", "2511"), ("Tilburg", "5038"), ("Breda", "4811")),
        ("Hoofdstraat", "Kerkstraat", "Industrieweg", "Stationsplein", "Dorpsstraat", "Havenkade"),
        ("B.V.", "N.V.", "V.O.F.", "B.V."),
        "& Zonen",
        ("Techniek", "Transport", "Installatie", "Groothandel", "Schoonmaak"),
        ("Sanne", "Daan", "Lotte", "Bram", "Eva", "Thijs"),
    ),
    "es": Market(
        "es", "Spain", "es",
        ("García", "Fernández", "López", "Martínez", "Sánchez", "Pérez", "Gómez", "Ruiz", "Hernández", "Díaz",
         "Moreno", "Muñoz", "Álvarez", "Romero"),
        ("Levante", "Castilla", "Andalucía", "Cantábrico", "Mediterráneo"),
        (("Madrid", "280"), ("Barcelona", "080"), ("Valencia", "460"), ("Sevilla", "410"), ("Zaragoza", "500"),
         ("Bilbao", "480"), ("Málaga", "290")),
        ("Calle Mayor", "Avenida de la Constitución", "Paseo de la Castellana", "Calle del Mar", "Plaza de España"),
        ("S.L.", "S.A.", "S.L.U."),
        "e Hijos",
        ("Suministros", "Servicios", "Transportes", "Informática", "Construcciones", "Limpieza"),
        ("Lucía", "Pablo", "Carmen", "Javier", "Marta", "Álvaro"),
    ),
    "it": Market(
        "it", "Italy", "it",
        ("Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Colombo", "Ricci", "Marino", "Greco", "Bruno",
         "Gallo", "D'Amico"),
        ("Lombardia", "Adriatica", "Toscana", "Ligure", "Veneta"),
        (("Milano", "201"), ("Roma", "001"), ("Torino", "101"), ("Bologna", "401"), ("Napoli", "801"),
         ("Firenze", "501"), ("Verona", "371")),
        ("Via Roma", "Via Garibaldi", "Corso Italia", "Via Verdi", "Piazza Duomo", "Via Mazzini"),
        ("S.r.l.", "S.p.A.", "S.n.c."),
        "& Figli",
        ("Forniture", "Servizi", "Trasporti", "Informatica", "Costruzioni", "Tipografia"),
        ("Giulia", "Marco", "Chiara", "Luca", "Francesca", "Matteo"),
    ),
    "uk": Market(
        "uk", "United Kingdom", "co.uk",
        ("Smith", "Jones", "Taylor", "Brown", "Wilson", "Evans", "Thomas", "Roberts", "Walker", "Wright", "Robinson",
         "Thompson", "White", "Hughes", "Edwards", "Green", "Hall", "Wood", "Harris", "Clarke", "Patel", "Jackson",
         "Turner", "O'Connor"),
        ("Westbury", "Oakleigh", "Kingsway", "Thames Valley", "Pennine", "Cotswold", "Albion", "Hartley", "Ashford",
         "Millbrook", "Northumbria"),
        (("London", "EC1A"), ("Manchester", "M1"), ("Birmingham", "B2"), ("Leeds", "LS1"), ("Bristol", "BS1"),
         ("Glasgow", "G2"), ("Edinburgh", "EH2"), ("Cardiff", "CF10"), ("Sheffield", "S1"),
         ("Newcastle upon Tyne", "NE1"), ("Nottingham", "NG1"), ("Reading", "RG1")),
        ("High Street", "Station Road", "Church Lane", "Victoria Road", "Mill Lane", "King Street", "Queens Road",
         "Park Road", "London Road", "The Green"),
        ("Ltd", "Limited", "Ltd", "PLC", "LLP", "Ltd"),
        "& Sons",
    ),
    "in": Market(
        "in", "India", "in",
        ("Sharma", "Patel", "Kapoor", "Gupta", "Mehta", "Iyer", "Reddy", "Nair", "Singh", "Joshi", "Khanna", "Bose",
         "Chatterjee", "Desai", "Agarwal", "Verma", "Rao", "Malhotra", "Banerjee"),
        ("Bharat", "Deccan", "Konkan", "Malabar", "Himalaya", "Ganga", "Sahyadri", "Kaveri"),
        (("Mumbai, Maharashtra", "400"), ("Bengaluru, Karnataka", "560"), ("Chennai, Tamil Nadu", "600"),
         ("New Delhi, Delhi", "110"), ("Hyderabad, Telangana", "500"), ("Pune, Maharashtra", "411"),
         ("Ahmedabad, Gujarat", "380"), ("Kolkata, West Bengal", "700"), ("Jaipur, Rajasthan", "302"),
         ("Surat, Gujarat", "395")),
        ("Plot 45, MIDC Andheri East", "12, Residency Road", "Shop No. 7, Linking Road", "3rd Floor, Nehru Place",
         "14/2, Industrial Estate", "Survey No. 88, Hinjewadi Phase 2", "27, Mount Road"),
        ("Pvt. Ltd.", "Private Limited", "Limited", "Enterprises", "Traders", "Industries", "Pvt. Ltd."),
        "& Sons",
        (),
        ("Aarav", "Ananya", "Vikram", "Meera", "Rohan", "Kavya"),
    ),
    "ca": Market(
        "ca", "Canada", "ca",
        ("Tremblay", "Roy", "Gagnon", "Côté", "Bouchard", "Lavoie", "Fortin", "Gauthier", "Morin", "Pelletier",
         "MacDonald", "Campbell", "Wilson", "Singh", "Chen"),
        ("Maple Leaf", "Northern", "Prairie", "Laurentian", "Harbourfront", "Rideau", "Cascade", "Great Lakes",
         "Pacific"),
        (("Toronto, ON", "M5H"), ("Montréal, QC", "H3B"), ("Vancouver, BC", "V6B"), ("Calgary, AB", "T2P"),
         ("Ottawa, ON", "K1P"), ("Edmonton, AB", "T5J"), ("Winnipeg, MB", "R3C"), ("Halifax, NS", "B3J")),
        ("Main St", "King St W", "Rue Sainte-Catherine", "Granville St", "Yonge St", "Portage Ave", "Rue Saint-Denis"),
        ("Inc.", "Ltd.", "Inc.", "Ltée"),
        "& Sons",
    ),
    "au": Market(
        "au", "Australia", "com.au",
        ("Smith", "Williams", "Brown", "Wilson", "Taylor", "Anderson", "Thompson", "Walker", "White", "Harris",
         "Martin", "Jackson", "Clarke", "Nguyen", "Campbell"),
        ("Tasman", "Coral Coast", "Outback", "Harbour City", "Southern Cross", "Bluegum", "Wattle", "Murray"),
        (("Sydney NSW", "2000"), ("Melbourne VIC", "3000"), ("Brisbane QLD", "4000"), ("Perth WA", "6000"),
         ("Adelaide SA", "5000"), ("Canberra ACT", "2600"), ("Hobart TAS", "7000"), ("Gold Coast QLD", "4217")),
        ("George St", "Collins St", "Flinders Lane", "Queen St", "Hay St", "Pitt St", "King William St"),
        ("Pty Ltd", "Pty. Ltd.", "Pty Ltd", "Ltd", "& Co"),
        "& Sons",
    ),
}
# fmt: on

# Industry -> (words for company names, item catalogue). Item line: description|unit|price range|quantity range.
INDUSTRIES: dict[str, tuple[tuple[str, ...], str]] = {
    "it": (
        ("Systems", "Technologies", "Computing", "Networks", "Digital", "IT Solutions", "Data"),
        """24-inch LED monitor, 1080p|pcs|120-260|1-12
Wireless keyboard and mouse set|pcs|25-70|1-30
USB-C docking station, 11-in-1|pcs|65-190|1-15
16 GB DDR4 SODIMM memory module|pcs|32-75|2-40
500 GB NVMe SSD|pcs|45-110|1-30
Cat6 patch cable, 3 m (pack of 10)|pk|14-35|1-20
Laptop 15.6", 16 GB RAM, 512 GB SSD|pcs|620-1450|1-10
24-port Gigabit managed switch|pcs|140-390|1-5
UPS 1500 VA line-interactive|pcs|130-310|1-6
Wi-Fi 6 access point, ceiling mount|pcs|85-210|1-12
HDMI 2.0 cable, 2 m|pcs|6-18|2-50
1080p webcam with microphone|pcs|28-85|1-25
Toner cartridge, black, high yield|pcs|55-140|1-20
Extended warranty, 3 years (per device)|pcs|45-160|1-30
On-site installation and configuration|hrs|65-120|2-16""",
    ),
    "software": (
        ("Software", "Cloud", "Labs", "Analytics", "Apps", "Digital"),
        """Annual licence - Team plan (per seat)|seat|96-420|3-60
Premium support - monthly|mo|150-900|1-12
Cloud hosting - 4 vCPU / 16 GB instance|mo|120-340|1-6
Object storage, 1 TB per month|TB|18-45|1-30
API usage overage (per 1,000 calls)|k|0.4-2.5|50-900
Implementation workshop (per day)|days|900-1800|1-5
SSL certificate, 1 year|pcs|45-210|1-8
Domain renewal (.com), 1 year|pcs|11-22|1-20
Data migration - fixed fee|lot|1200-6500|1-1
Security audit - per application|pcs|1800-5200|1-4
Single sign-on add-on (per user per month)|usr|2-9|10-250
Training session, remote (up to 10 attendees)|pcs|350-950|1-6""",
    ),
    "consulting": (
        ("Consulting", "Advisory", "Partners", "Strategy", "Associates"),
        """Senior consultant, hourly rate|hrs|95-210|4-80
Project management - March engagement, weekly steering meetings and status reporting|hrs|85-160|8-60
Business analysis workshop series - requirements gathering, stakeholder interviews and documentation|days|700-1400|1-6
Architecture review and recommendations report - fixed fee|lot|2400-9500|1-1
Travel expenses - client site visit (flights, hotel, per diem)|lot|180-1650|1-4
Quarterly strategy report - market sizing, competitor benchmarking and executive summary|lot|1800-6200|1-3
Process mapping and optimisation, discovery phase|days|650-1200|2-12
Change management support - communications plan and training materials|hrs|75-140|6-50
Data analysis and dashboard build|hrs|90-170|8-48
Interim management services, monthly retainer|mo|3800-9800|1-6""",
    ),
    "office": (
        ("Office Supplies", "Stationery", "Business Supplies", "Office Solutions"),
        """A4 copy paper 80 gsm, 5 reams per box|box|18-34|2-60
Ballpoint pens, blue (box of 50)|box|7-16|1-20
Stapler, heavy duty, 100-sheet|pcs|14-35|1-12
Whiteboard markers, assorted (pack of 12)|pk|8-17|1-20
Ring binders A4, 4 cm (pack of 10)|pk|13-28|1-25
Sticky notes 76 x 76 mm (pack of 12)|pk|9-19|2-40
Ergonomic office chair, mesh back|pcs|120-340|1-20
Desk organiser, bamboo|pcs|12-30|1-15
Laminating pouches A4 (pack of 100)|pk|10-24|1-10
Shredder, cross-cut, 12 sheets|pcs|65-190|1-4
Hanging file folders (box of 25)|box|13-26|1-30
Coffee, ground, 1 kg|pcs|9-21|2-20""",
    ),
    "construction": (
        ("Builders", "Construction", "Building Supplies", "Civil Works", "Aggregates"),
        """Portland cement, 25 kg bag|bag|5.5-9.8|20-400
Steel rebar 12 mm x 6 m|pcs|9-16|50-600
Plywood 18 mm, 2440 x 1220 mm sheet|pcs|38-72|10-120
Excavator hire, 8 t - per day|days|280-460|1-14
Skilled labour - per hour|hrs|32-68|8-160
Scaffolding rental - per week|wk|140-420|1-12
Ready-mix concrete C25/30 per m3|m3|95-150|3-60
Safety helmets (box of 10)|box|45-90|1-8
Gravel 20 mm, bulk bag|pcs|38-65|2-40
Roofing membrane, 1 x 20 m roll|roll|58-120|2-30
Damp-proof course 150 mm x 30 m|roll|18-36|2-20
Skip hire, 8 yard|pcs|180-290|1-6""",
    ),
    "food": (
        ("Catering", "Foods", "Kitchen", "Provisions", "Fine Foods"),
        """Catering - lunch buffet (per person)|pp|11-28|20-250
Coffee and tea service - per day|days|85-190|1-10
Fresh produce box, mixed|box|24-48|2-40
Bottled water 500 ml (case of 24)|case|6-12|5-120
Bakery assortment, 12 pieces|box|14-30|2-30
Event staff - per hour|hrs|22-45|8-120
Dessert platter, serves 10|pcs|36-85|1-12
Sandwich platter, serves 12|pcs|48-95|1-14
Soft drinks assorted (case of 24)|case|11-20|2-40
Disposable tableware pack (100 pieces)|pk|9-22|2-20
Equipment hire - chafing dishes|set|18-45|1-10""",
    ),
    "medical": (
        ("Medical", "Healthcare Supplies", "Pharma", "Medical Supplies"),
        """Nitrile gloves, medium (box of 100)|box|6-14|10-300
Surgical masks type IIR (box of 50)|box|4-11|10-200
Hand sanitiser 500 ml, pump|pcs|3-8|10-120
Digital thermometer, infrared|pcs|14-40|2-30
Blood pressure monitor, upper arm|pcs|28-75|1-12
First aid kit, workplace, 50 persons|pcs|36-95|1-15
Disposable syringes 5 ml (pack of 100)|pk|8-19|2-60
Couch paper roll 50 cm x 50 m|pcs|3.5-8|10-100
Sterile dressings assorted (box of 50)|box|7-16|2-40
Pulse oximeter, fingertip|pcs|16-45|2-20
Sharps container 4 L|pcs|2.8-7|10-80""",
    ),
    "logistics": (
        ("Logistics", "Freight", "Transport", "Shipping", "Haulage"),
        """Freight - LTL pallet, Chicago to Dallas|plt|180-520|1-24
Container handling charge (per container)|pcs|95-240|1-12
Customs clearance fee, per consignment|pcs|65-180|1-10
Warehouse storage - per pallet per month|plt|9-22|10-200
Courier - express parcel, next day|pcs|7-28|5-120
Fuel surcharge|lot|45-320|1-1
Last-mile delivery, residential|pcs|9-26|5-90
Pallet wrapping and labelling|plt|3.5-9|10-120
Cross-dock handling per pallet|plt|6-15|10-80
Temperature-controlled transport, per km|km|1.4-2.9|80-1400""",
    ),
    "print": (
        ("Print", "Graphics", "Media", "Creative", "Design Studio"),
        """Business cards, 500 pcs, matt laminated|pcs|24-58|1-10
A5 flyers, 1,000 pcs, 150 gsm gloss|pcs|65-135|1-8
Roll-up banner 85 x 200 cm with stand|pcs|45-110|1-12
Brochure design and layout|hrs|55-110|4-40
Social media campaign management - monthly|mo|650-2400|1-6
Logo redesign - three concepts|lot|450-1800|1-1
Product photography, half day|pcs|320-780|1-4
Large-format poster A1 on foam board|pcs|18-42|2-30
Email newsletter design and send|pcs|160-420|1-6
Trade show display, modular 3 x 3 m|pcs|1200-2800|1-2""",
    ),
    "cleaning": (
        ("Cleaning Services", "Facilities", "Maintenance", "Property Services"),
        """Office cleaning - weekly service, 3 visits|wk|180-520|1-12
Window cleaning - quarterly, external glazing|pcs|140-420|1-4
HVAC maintenance visit and filter change|pcs|180-420|1-6
Replacement filter set, G4/F7|set|35-95|1-12
Pest control treatment, per visit|pcs|85-190|1-6
Waste collection - monthly, 1,100 L bin|mo|38-95|1-12
Carpet deep clean, per m2|m2|2.5-6|40-500
Washroom consumables restock|lot|65-190|1-6
Deep clean - kitchen and canteen|pcs|220-480|1-3""",
    ),
    "industrial": (
        ("Industrial Supply", "Engineering", "Fasteners", "Hydraulics", "Tools"),
        """Ball bearing 6205-2RS|pcs|3.5-9|20-300
V-belt A42|pcs|6-14|5-60
Hydraulic hose 1/2" x 2 m, with fittings|pcs|28-65|2-40
Stainless steel bolts M8 x 40 (box of 100)|box|14-32|2-30
Welding wire ER70S-6, 15 kg spool|pcs|38-72|1-12
Safety gloves, cut level 5 (pair)|pr|3.8-9|10-120
Lithium grease 400 g cartridge|pcs|3.2-7.5|10-100
Industrial fan 24", wall mount|pcs|85-190|1-8
Pneumatic fitting set, 1/4" (50 pcs)|set|24-52|1-20
Proximity sensor M12 NPN|pcs|11-32|2-40
Cutting disc 125 mm (box of 25)|box|16-35|2-24
Pressure gauge 0-10 bar|pcs|9-26|2-30""",
    ),
    "legal": (
        ("Accounting", "Legal", "Chartered Accountants", "Tax Advisers", "Solicitors"),
        """Bookkeeping services - monthly|mo|220-780|1-3
Annual accounts preparation and filing|lot|950-3800|1-1
Quarterly VAT return preparation and submission|lot|180-540|1-4
Legal consultation|hrs|150-380|1-20
Contract review and redline, supplier agreement|lot|450-2200|1-3
Payroll processing (per employee per month)|emp|6-18|5-80
Company registration filing fee|lot|120-480|1-2
Tax advisory - corporate restructuring options|hrs|165-340|2-24
Statutory audit support - fieldwork|days|600-1100|1-8""",
    ),
    "travel": (
        ("Hospitality", "Events", "Travel", "Conference Services"),
        """Hotel accommodation, standard room (per night)|night|85-240|1-14
Conference room hire - full day, up to 20 people|days|280-720|1-4
Airport transfer, executive saloon|pcs|55-140|1-6
Event registration fee|pcs|180-650|1-12
Meals and incidentals, per diem|days|38-85|1-14
Coach hire with driver, 49 seats - per day|days|520-980|1-3
AV equipment rental - projector and PA|pcs|120-340|1-4
Welcome reception - canapes and drinks (per head)|pp|24-60|20-120""",
    ),
}

# Receipt stores: (store words, items)
STORES: dict[str, tuple[tuple[str, ...], str]] = {
    "hardware": (
        ("Hardware", "Home Centre", "Building Supply", "Tool & Lumber"),
        """Wood screws 3" (box of 100)|ea|6-14|1-4
Interior paint 3.78 L eggshell|ea|32-58|1-4
Paint roller kit 9"|ea|9-19|1-3
LED bulb 60W equiv. (4 pack)|ea|7-15|1-4
Extension cord 15 m outdoor|ea|19-36|1-1
Work gloves, leather|ea|9-22|1-2
Duct tape 48 mm x 25 m|ea|5-12|1-4
Garden hose 25 m|ea|24-48|1-1
Hammer 16 oz|ea|14-28|1-1
Silicone sealant, clear|ea|6-13|1-4
Batteries AA (24 pack)|ea|13-24|1-2
Plywood sheet 3/4" 4x8|ea|54-86|1-3
Measuring tape 8 m|ea|8-17|1-1
Sandpaper assorted (20 pk)|ea|6-13|1-3""",
    ),
    "cafe": (
        ("Café", "Coffee House", "Bakery", "Roasters"),
        """Latte 16 oz|ea|4.5-6.5|1-3
Americano 12 oz|ea|3.2-4.5|1-3
Blueberry muffin|ea|3.2-4.6|1-4
Croissant, butter|ea|3.4-4.8|1-4
Iced coffee 20 oz|ea|4.4-5.8|1-3
Turkey club sandwich|ea|9.5-13.5|1-3
Soup of the day, bowl|ea|6.2-8.8|1-2
Cappuccino 12 oz|ea|4.2-5.6|1-3
Chai latte 16 oz|ea|4.8-6.2|1-2
Cookie, chocolate chunk|ea|2.6-3.6|1-4
Bagel with cream cheese|ea|4.6-6.4|1-3
Bottled water 500 ml|ea|1.9-3.2|1-3""",
    ),
    "pharmacy": (
        ("Pharmacy", "Drugstore", "Health & Beauty"),
        """Vitamin D3 1000 IU (180 caps)|ea|11-21|1-2
Ibuprofen 200 mg (100 tablets)|ea|9-16|1-2
Hand cream 100 mL|ea|5.5-11|1-2
Toothpaste 120 mL|ea|3.2-6.4|1-3
Allergy relief 24 h (30 tabs)|ea|14-25|1-1
Bandages assorted (60 pack)|ea|5.5-9.8|1-2
Shampoo 400 mL|ea|6.5-12.5|1-2
Sunscreen SPF 50, 200 mL|ea|12-22|1-2
Cough syrup 250 mL|ea|8-15|1-1
Lip balm (2 pack)|ea|3.6-6.5|1-3
Facial tissue (3 boxes)|ea|4.2-7.5|1-2""",
    ),
    "books": (
        ("Books", "Bookshop", "Books & Stationery"),
        """Paperback novel, bestseller|ea|12-22|1-3
Hardcover cookbook|ea|28-48|1-1
Notebook, A5 lined|ea|6-14|1-4
Fountain pen ink|ea|9-18|1-1
Greeting card|ea|4-8|1-4
Wall calendar|ea|12-22|1-1
Bookmarks (set of 6)|ea|4-9|1-2
Children's picture book|ea|9-17|1-3""",
    ),
    "pets": (
        ("Pet Supplies", "Pet Centre", "Pet Food & Supply"),
        """Dry dog food 12 kg|ea|42-76|1-1
Cat litter 18 L|ea|14-26|1-2
Dog chew toy|ea|6-14|1-3
Cat treats 85 g|ea|2.2-4.5|1-6
Leash 1.8 m, nylon|ea|11-24|1-1
Fish flakes 100 g|ea|5-9|1-2
Pet shampoo 500 mL|ea|9-17|1-1
Bird seed 2 kg|ea|7-13|1-2""",
    ),
}

NOTES = (
    "Payment is due within {days} days of the invoice date. "
    "Late payments are subject to a 1.5% monthly service charge.",
    "Thank you for your business. Please quote the invoice number with your payment.",
    "Goods remain the property of the supplier until paid in full.",
    "Bank transfer preferred. Please allow 2-3 working days for funds to clear.",
    "This is a computer-generated invoice and does not require a signature.",
    "Queries about this invoice must be raised within 14 days of receipt (ref. order 2024-{n4}).",
    "Prices are valid for 30 days. Delivery within 5-7 business days after order confirmation.",
    "Returns accepted within 30 days in original packaging. A 15% restocking fee may apply.",
    "Please make cheques payable to the company name shown above. Customer account {n5}.",
    "Early settlement discount of 2% if paid within 10 days (not applied to this invoice).",
)


@dataclass(frozen=True)
class ItemTemplate:
    description: str
    unit: str
    price: tuple[float, float]
    qty: tuple[int, int]


def _parse_items(blob: str) -> tuple[ItemTemplate, ...]:
    out: list[ItemTemplate] = []
    for line in blob.splitlines():
        desc, unit, price, qty = line.strip().split("|")
        p0, p1 = price.split("-")
        q0, q1 = qty.split("-")
        out.append(ItemTemplate(desc, unit, (float(p0), float(p1)), (int(q0), int(q1))))
    return tuple(out)


INDUSTRY_ITEMS = {k: _parse_items(v[1]) for k, v in INDUSTRIES.items()}
STORE_ITEMS = {k: _parse_items(v[1]) for k, v in STORES.items()}


# ====================================================================================================================
# spec types
# ====================================================================================================================
@dataclass
class Party:
    name: str | None
    lines: list[str]
    phone: str | None = None
    email: str | None = None
    web: str | None = None
    ids: list[tuple[str, str]] = field(default_factory=list)  # (label, value) tax / registration identifiers
    bank: list[tuple[str, str]] = field(default_factory=list)  # (label, value) payment details
    contact: str | None = None
    state_code: str | None = None  # India: GST state code of the address


@dataclass
class Item:
    description: str
    quantity: Decimal
    unit_price: Decimal
    unit: str
    amount: Decimal = Decimal(0)
    price_printed: bool = True
    vat_class: int = 0
    hsn: str | None = None


@dataclass
class TaxLine:
    kind: str  # tax | vat | gst | cgst | sgst | igst | state | county | pst | hst
    rate: Decimal
    amount: Decimal = Decimal(0)
    base: Decimal = Decimal(0)


@dataclass
class ErrorSpec:
    kind: str  # TOTAL_MISMATCH | LINE_ITEMS_SUM_MISMATCH | LINE_ITEM_MATH
    variant: str
    delta: Decimal = Decimal(0)
    index: int = 0


@dataclass
class InvoiceSpec:
    uid: str
    template: str
    doc_type: DocumentType
    market: str
    currency: str
    currency_printed: bool
    invoice_number: str | None
    invoice_date: date | None
    due_date: date | None
    service_date: date | None
    supplier: Party
    customer: Party | None
    ship_to: Party | None
    items: list[Item]
    # inputs of the arithmetic
    tax_scheme: str  # none | single | pair | classes | gst_split | igst
    tax_rates: list[Decimal]
    discount_pct: Decimal | None
    discount_fixed: Decimal | None
    shipping_amount: Decimal | None
    combined_tax_row: bool  # India: print "Total GST @ 18%" under CGST/SGST
    rate_printed: bool  # the tax line states its percentage
    omit: frozenset[str]
    error: ErrorSpec | None
    extra: dict[str, Any]
    style: dict[str, Any] = field(default_factory=dict)
    # printed values, filled by finalize()
    subtotal: Decimal | None = None
    discount: Decimal | None = None
    shipping: Decimal | None = None
    taxes: list[TaxLine] = field(default_factory=list)
    total: Decimal | None = None
    dropped: int | None = None

    def shows(self, name: str) -> bool:
        return name not in self.omit

    @property
    def rows(self) -> list[Item]:
        """The line items actually printed (an intentional error may drop one row)."""
        if "line_items" in self.omit:
            return []
        if self.dropped is None or not (0 <= self.dropped < len(self.items)):
            return self.items
        return [it for i, it in enumerate(self.items) if i != self.dropped]

    @property
    def tax_total(self) -> Decimal | None:
        return sum((t.amount for t in self.taxes), Decimal(0)) if self.taxes else None

    @property
    def tax_rate(self) -> Decimal | None:
        """The single overall rate printed, if any (see ``eval.labels``)."""
        if len(self.taxes) == 1:
            return plain(self.taxes[0].rate) if self.rate_printed else None
        if self.tax_scheme == "gst_split" and self.combined_tax_row and self.taxes:
            return plain(sum((t.rate for t in self.taxes), Decimal(0)))
        return None

    def record(self) -> LabeledRecord:
        """The ground-truth label of this document (location fields are filled by the builder)."""
        items = [
            LineItem(
                description=it.description,
                quantity=plain(it.quantity),
                unit_price=q2(it.unit_price) if it.price_printed else None,
                amount=q2(it.amount),
            )
            for it in self.rows
        ]
        tax_total = self.tax_total
        return LabeledRecord(
            invoice_number=self.invoice_number,
            invoice_date=self.invoice_date,
            due_date=self.due_date,
            supplier_name=self.supplier.name,
            customer_name=self.customer.name if self.customer else None,
            subtotal=self.subtotal,
            discount=self.discount,
            shipping=self.shipping,
            tax=q2(tax_total) if tax_total is not None else None,
            tax_rate=self.tax_rate,
            total=self.total,
            currency=self.currency if self.currency_printed else None,
            document_type=self.doc_type,
            line_items=items,
        )


# ====================================================================================================================
# helpers
# ====================================================================================================================
def ascii_slug(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")


def digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(n))


def group(text: str, size: int) -> str:
    return " ".join(text[i : i + size] for i in range(0, len(text), size))


def random_date(rng: random.Random, start: date = date(2023, 1, 2), end: date = date(2025, 12, 19)) -> date:
    span = (end - start).days
    for _ in range(8):
        day = start + timedelta(days=rng.randint(0, span))
        if day.weekday() < 5 or rng.random() < 0.15:
            return day
    return day


def financial_year(day: date) -> str:
    start = day.year if day.month >= 4 else day.year - 1
    return f"{start % 100:02d}-{(start + 1) % 100:02d}"


# --------------------------------------------------------------------------------------------------------------------
# parties
# --------------------------------------------------------------------------------------------------------------------
def company_name(rng: random.Random, market: Market, words: tuple[str, ...], *, long: bool = False) -> str:
    word = rng.choice(market.local_words) if market.local_words and rng.random() < 0.45 else rng.choice(words)
    suffix = rng.choice(market.suffixes)
    surname, other = rng.sample(market.surnames, 2)
    place = rng.choice(market.places)
    if long:
        return f"{place} {surname} International {word} and Distribution Services {suffix}"
    patterns = (
        (3, f"{surname} {word} {suffix}"),
        (3, f"{place} {word} {suffix}"),
        (2, f"{surname} & {other} {word}"),
        (2, f"{surname} & {other} {suffix}"),
        (2, f"{surname} {market.sons} {suffix}"),
        (1, f"The {place} {word} Company") if market.key in {"us", "uk", "au"} else (1, f"{place} {word} {suffix}"),
        (1, f"{surname} {word}"),
    )
    name = rng.choices([p for _, p in patterns], weights=[w for w, _ in patterns])[0]
    if market.key == "fr" and rng.random() < 0.3:
        name = f"Établissements {surname} {market.sons}"
    if market.key == "es" and rng.random() < 0.3:
        name = f"Hermanos {surname} y Cía {suffix}"
    return name.upper() if rng.random() < 0.12 else name


def phone(rng: random.Random, market: Market) -> str:
    k = market.key
    if k == "us":
        return rng.choice(
            (
                f"+1 ({rng.choice(('212', '312', '415', '512', '617', '713', '206', '303'))}) 555-01{digits(rng, 2)}",
                f"({rng.choice(('212', '312', '415', '512'))}) 555-01{digits(rng, 2)}",
            )
        )
    if k == "ca":
        return f"({rng.choice(('514', '416', '604', '403', '613'))}) 555-01{digits(rng, 2)}"
    if k == "de":
        return f"+49 {rng.choice(('30', '40', '89', '221', '69', '711'))} {digits(rng, 7)}"
    if k == "fr":
        return f"+33 {rng.randint(1, 5)} {digits(rng, 2)} {digits(rng, 2)} {digits(rng, 2)} {digits(rng, 2)}"
    if k == "nl":
        return f"+31 {rng.choice(('20', '10', '30', '40'))} {digits(rng, 3)} {digits(rng, 4)}"
    if k == "es":
        return f"+34 {rng.choice(('91', '93', '96', '95'))} {digits(rng, 3)} {digits(rng, 2)} {digits(rng, 2)}"
    if k == "it":
        return f"+39 {rng.choice(('02', '06', '011', '051'))} {digits(rng, 4)} {digits(rng, 4)}"
    if k == "uk":
        return rng.choice(
            (f"+44 20 7946 0{digits(rng, 3)}", f"0161 496 0{digits(rng, 3)}", f"0113 496 0{digits(rng, 3)}")
        )
    if k == "in":
        return rng.choice(
            (
                f"+91 {rng.choice(('22', '80', '44', '11', '40'))} {digits(rng, 4)} {digits(rng, 4)}",
                f"+91 9{digits(rng, 4)} {digits(rng, 5)}",
            )
        )
    return rng.choice(
        (
            f"+61 {rng.choice(('2', '3', '7', '8'))} {digits(rng, 4)} {digits(rng, 4)}",
            f"(0{rng.choice(('2', '3', '7'))}) {digits(rng, 4)} {digits(rng, 4)}",
        )
    )


GST_STATES = {
    "Maharashtra": "27",
    "Karnataka": "29",
    "Tamil Nadu": "33",
    "Delhi": "07",
    "Telangana": "36",
    "Gujarat": "24",
    "West Bengal": "19",
    "Rajasthan": "08",
}


def tax_ids(rng: random.Random, market: Market, state_code: str | None) -> list[tuple[str, str]]:
    k = market.key
    if k == "us":
        return [("EIN", f"{digits(rng, 2)}-{digits(rng, 7)}")]
    if k == "ca":
        return [("GST/HST No.", f"{digits(rng, 9)} RT{digits(rng, 4)}")]
    if k == "de":
        ids = [("USt-IdNr.", f"DE{digits(rng, 9)}")]
        return ids + (
            [("Steuer-Nr.", f"{digits(rng, 2)}/{digits(rng, 3)}/{digits(rng, 5)}")] if rng.random() < 0.4 else []
        )
    if k == "fr":
        ids = [("TVA", f"FR{digits(rng, 2)} {digits(rng, 9)}")]
        return ids + (
            [("SIRET", f"{digits(rng, 3)} {digits(rng, 3)} {digits(rng, 3)} {digits(rng, 5)}")]
            if rng.random() < 0.5
            else []
        )
    if k == "nl":
        return [("BTW", f"NL{digits(rng, 9)}B{digits(rng, 2)}"), ("KvK", digits(rng, 8))]
    if k == "es":
        return [("NIF", f"ES{letters(rng, 1)}{digits(rng, 8)}")]
    if k == "it":
        return [("P.IVA", f"IT{digits(rng, 11)}")]
    if k == "uk":
        ids = [("VAT Reg. No.", f"GB {digits(rng, 3)} {digits(rng, 4)} {digits(rng, 2)}")]
        return ids + ([("Company No.", f"0{digits(rng, 7)}")] if rng.random() < 0.6 else [])
    if k == "in":
        pan = f"{letters(rng, 5)}{digits(rng, 4)}{letters(rng, 1)}"
        return [
            ("GSTIN", f"{state_code or '27'}{pan}1Z{rng.choice('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')}"),
            ("PAN", pan),
        ]
    return [("ABN", f"{digits(rng, 2)} {digits(rng, 3)} {digits(rng, 3)} {digits(rng, 3)}")]


def bank_details(rng: random.Random, market: Market) -> list[tuple[str, str]]:
    k = market.key
    if k == "de":
        return [("IBAN", group(f"DE{digits(rng, 20)}", 4)), ("BIC", f"{letters(rng, 4)}DE{letters(rng, 2)}XXX")]
    if k == "fr":
        return [("IBAN", group(f"FR{digits(rng, 25)}", 4)), ("BIC", f"{letters(rng, 4)}FR{letters(rng, 2)}XXX")]
    if k == "nl":
        return [
            ("IBAN", group(f"NL{digits(rng, 2)}{letters(rng, 4)}{digits(rng, 10)}", 4)),
            ("BIC", f"{letters(rng, 4)}NL2A"),
        ]
    if k == "es":
        return [("IBAN", group(f"ES{digits(rng, 22)}", 4)), ("BIC", f"{letters(rng, 4)}ES{letters(rng, 2)}")]
    if k == "it":
        return [
            ("IBAN", group(f"IT{digits(rng, 2)}{letters(rng, 1)}{digits(rng, 22)}", 4)),
            ("BIC", f"{letters(rng, 4)}IT{letters(rng, 2)}"),
        ]
    if k == "uk":
        return [("Sort Code", f"{digits(rng, 2)}-{digits(rng, 2)}-{digits(rng, 2)}"), ("Account No.", digits(rng, 8))]
    if k == "us":
        return [("ACH routing", digits(rng, 9)), ("Account", digits(rng, 12))]
    if k == "in":
        return [("A/c No.", digits(rng, 12)), ("IFSC", f"{letters(rng, 4)}0{digits(rng, 6)}")]
    if k == "au":
        return [("BSB", f"{digits(rng, 3)}-{digits(rng, 3)}"), ("Account", f"{digits(rng, 4)} {digits(rng, 4)}")]
    return []


def address(rng: random.Random, market: Market) -> tuple[list[str], str | None]:
    city, region = rng.choice(market.cities)
    street = rng.choice(market.streets)
    k = market.key
    n = rng.randint(1, 240)
    if k == "us":
        suite = f", Suite {rng.randint(100, 950)}" if rng.random() < 0.4 else ""
        town, state = city.split(", ")
        return [f"{n} {street}{suite}", f"{town}, {state} {region}{digits(rng, 2)}"], None
    if k == "ca":
        town, prov = city.split(", ")
        return [
            f"{n} {street}",
            f"{town}, {prov} {region} {rng.randint(1, 9)}{letters(rng, 1)}{rng.randint(1, 9)}",
        ], None
    if k == "de":
        return [f"{street} {n}", f"{region}{digits(rng, 2)} {city}"], None
    if k == "fr":
        return [f"{n} {street}", f"{region}{digits(rng, 2)} {city}"], None
    if k == "nl":
        return [f"{street} {n}", f"{region} {letters(rng, 2)} {city}"], None
    if k == "es":
        return [f"{street}, {n}", f"{region}{digits(rng, 2)} {city}"], None
    if k == "it":
        return [f"{street}, {n}", f"{region}{digits(rng, 2)} {city}"], None
    if k == "uk":
        return [f"{n} {street}", f"{city} {region} {rng.randint(1, 9)}{letters(rng, 2)}"], None
    if k == "in":
        town, state = city.split(", ")
        return [street, f"{town} {region}{digits(rng, 3)}, {state}"], GST_STATES.get(state)
    town, state = city.rsplit(" ", 1)
    return [f"Level {rng.randint(2, 30)}, {n} {street}", f"{town} {state} {region}"], None


def make_party(
    rng: random.Random,
    market: Market,
    words: tuple[str, ...],
    *,
    long: bool = False,
    person: bool = False,
    with_bank: bool = False,
    with_ids: bool = True,
) -> Party:
    lines, state_code = address(rng, market)
    if person:
        name = f"{rng.choice(market.first_names)} {rng.choice(market.surnames)}"
    else:
        name = company_name(rng, market, words, long=long)
    slug = ascii_slug(name)[:28].strip("-") or "company"
    party = Party(
        name=name,
        lines=[*lines, market.country] if market.key not in {"us", "uk"} or rng.random() < 0.35 else lines,
        phone=phone(rng, market),
        email=None
        if person
        else f"{rng.choice(('accounts', 'billing', 'info', 'office', 'sales'))}@{slug}.{market.tld}",
        web=None if person or rng.random() < 0.5 else f"www.{slug}.{market.tld}",
        ids=tax_ids(rng, market, state_code) if with_ids and not person else [],
        bank=bank_details(rng, market) if with_bank else [],
        state_code=state_code,
    )
    if not person and rng.random() < 0.3:
        party.contact = f"{rng.choice(market.first_names)} {rng.choice(market.surnames)}"
    return party


# --------------------------------------------------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------------------------------------------------
class NumberRegistry:
    """Hands out invoice numbers that are unique after normalisation (upper-case alphanumerics only)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    @staticmethod
    def normalise(number: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", number.upper())

    def claim(self, number: str) -> bool:
        key = self.normalise(number)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


NUMBER_PATTERNS: dict[str, tuple[str, ...]] = {
    "T1": ("INV-{yyyy}-{n4}", "INV-{n5}", "{pfx}-{n5}", "{n6}", "SO{n6}", "{yyyy}{mm}{n3}", "INV{n5}"),
    "T2": ("RE-{yyyy}-{n4}", "{yyyy}/{n4}", "RG{yy}{n5}", "FA-{yyyy}-{n5}", "{yyyy}-{n4}", "R{yy}{mm}-{n3}"),
    "T3": ("INV{n6}", "SI-{n5}", "{pfx}/{yy}/{n5}", "INV-{n4}", "UK{yy}{n5}", "{n5}"),
    "T4": ("{pfx}/{fy}/{n4}", "GST/{fy}/{n4}", "{pfx}{n5}", "INV/{yy}/{n4}", "TI-{n5}"),
    "T5": ("R-{n6}", "{n5}-{n3}", "{n7}", "{n4}-{n4}", "T{n6}"),
    "T6": ("INV-{n4}", "{yyyy}-{n3}", "AU-{n5}", "{pfx}{n4}", "{yyyy}{mm}-{n3}", "INV-{n5}"),
}


def make_number(
    rng: random.Random, template: str, day: date, supplier: str, registry: NumberRegistry, *, credit_note: bool = False
) -> str:
    pfx = "".join(w[0] for w in re.findall(r"[A-Za-z]+", ascii_slug(supplier).replace("-", " "))[:3]).upper() or "INV"
    for _ in range(50):
        pattern = rng.choice(NUMBER_PATTERNS[template])
        number = pattern.format(
            yyyy=day.year,
            yy=f"{day.year % 100:02d}",
            mm=f"{day.month:02d}",
            pfx=pfx,
            fy=financial_year(day),
            n3=digits(rng, 3),
            n4=digits(rng, 4),
            n5=digits(rng, 5),
            n6=digits(rng, 6),
            n7=digits(rng, 7),
        )
        if credit_note:
            number = rng.choice(("CN-", "CR", "GS-", "AV-")) + number
        if registry.claim(number):
            return number
    raise RuntimeError("could not find a unique invoice number")


# ====================================================================================================================
# per-template content profiles
# ====================================================================================================================
@dataclass(frozen=True)
class Profile:
    markets: tuple[str, ...]
    currencies: tuple[tuple[str, int], ...]
    due_probability: float
    discount_probability: float
    shipping_probability: float
    industries: tuple[str, ...]
    price_scale: tuple[float, float] = (1.0, 1.0)


PROFILES: dict[str, Profile] = {
    "T1": Profile(
        ("us",),
        (("USD", 88), ("CAD", 12)),
        0.88,
        0.16,
        0.26,
        ("it", "office", "industrial", "software", "print", "medical", "construction", "logistics"),
    ),
    "T2": Profile(
        ("de", "fr", "nl", "es", "it"),
        (("EUR", 100),),
        0.86,
        0.12,
        0.2,
        ("industrial", "it", "office", "cleaning", "print", "construction", "food", "legal"),
    ),
    "T3": Profile(
        ("uk",),
        (("GBP", 92), ("EUR", 8)),
        0.9,
        0.12,
        0.22,
        ("legal", "consulting", "office", "cleaning", "print", "it", "travel", "food"),
    ),
    "T4": Profile(
        ("in",),
        (("INR", 100),),
        0.86,
        0.14,
        0.2,
        ("industrial", "it", "construction", "office", "logistics", "medical", "print"),
        (62.0, 88.0),
    ),
    "T5": Profile(("ca",), (("CAD", 100),), 0.0, 0.0, 0.0, ("hardware", "cafe", "pharmacy", "books", "pets")),
    "T6": Profile(
        ("au",),
        (("AUD", 90), ("USD", 10)),
        0.95,
        0.14,
        0.22,
        ("consulting", "software", "print", "it", "travel", "food", "cleaning"),
    ),
}

US_RATES = ("4", "5", "5.5", "6", "6.25", "7", "7.25", "8", "8.25", "8.5", "8.875", "9", "10")
EU_RATES = {"de": ("19", "7"), "fr": ("20", "10", "5.5"), "nl": ("21", "9"), "es": ("21", "10"), "it": ("22", "10")}
GST_SLABS = (("5", "2.5"), ("12", "6"), ("18", "9"), ("18", "9"), ("18", "9"), ("28", "14"))
CA_TAX = (("single", ("5",)), ("pair", ("5", "7")), ("single", ("13",)), ("pair", ("5", "9.975")), ("single", ("15",)))


def _weighted(rng: random.Random, options: tuple[tuple[str, int], ...]) -> str:
    return rng.choices([o for o, _ in options], weights=[w for _, w in options])[0]


def _price(rng: random.Random, low: float, high: float, scale: tuple[float, float]) -> Decimal:
    value = rng.uniform(low, high) * rng.uniform(*scale)
    style = rng.random()
    if value >= 40:
        value = round(value) if style < 0.5 else round(value) - 0.01 if style < 0.65 else round(value * 2) / 2
    else:
        value = round(value * 20) / 20 if style < 0.4 else round(value, 2)
    return q2(Decimal(str(max(value, 0.5))))


def _quantity(rng: random.Random, template: ItemTemplate) -> Decimal:
    low, high = template.qty
    if template.unit == "hrs":
        return Decimal(str(rng.choice((rng.randint(low, high), rng.randint(low * 4, high * 4) / 4))))
    if template.unit in {"m2", "m3", "km", "TB"} and rng.random() < 0.3:
        return Decimal(str(round(rng.uniform(low, high), 1)))
    return Decimal(rng.randint(low, high))


LONG_CLAUSES = (
    "- includes preparation, on-site delivery and post-engagement follow-up as set out in statement of work SOW-{n}",
    "- covers all activities agreed in change request CR-{n}, including documentation, review meetings and hand-over "
    "to the operations team",
    "- time and materials under the framework agreement dated 14 June 2023 (ref. FA-{n}); travel and subsistence are "
    "invoiced separately",
)


def make_items(
    rng: random.Random,
    catalogue: tuple[ItemTemplate, ...],
    n: int,
    scale: tuple[float, float],
    *,
    max_desc: int,
    receipt: bool,
    template: str,
    long_text: bool = False,
) -> list[Item]:
    """Pick ``n`` distinct catalogue entries (a receipt never repeats a product) and price them."""
    pool = [t for t in catalogue if len(t.description) <= max_desc] or list(catalogue)
    if receipt:
        n = min(n, len(pool))
    chosen: list[ItemTemplate] = []
    while len(chosen) < n:  # distinct while possible; a long invoice repeats products with a variant code
        fresh = [t for t in pool if t not in chosen]
        chosen.append(rng.choice(fresh or pool))
    items: list[Item] = []
    seen: set[str] = set()
    for t in chosen:
        description = t.description
        if long_text:
            description += " " + rng.choice(LONG_CLAUSES).format(n=digits(rng, 4))
        elif not receipt and rng.random() < 0.22 and len(description) + 16 <= max_desc:
            description += f" (SKU {digits(rng, 5)}-{digits(rng, 4)})"
        if description in seen:
            description += f" - variant {rng.choice('ABCDEFGH')}{digits(rng, 2)}"
        seen.add(description)
        hsn = rng.choice(HSN_CODES) if template == "T4" else None
        items.append(Item(description, _quantity(rng, t), _price(rng, t.price[0], t.price[1], scale), t.unit, hsn=hsn))
    return items


HSN_CODES = ("8471", "9983", "7318", "3926", "8536", "4820", "9954", "9988", "2523", "3004", "8481")


def _item_count(rng: random.Random, n_items: int | None) -> int:
    if n_items is not None:
        return n_items
    band = rng.choices(((1, 3), (4, 8), (9, 15), (16, 25)), weights=(24, 42, 22, 12))[0]
    return rng.randint(*band)


# ====================================================================================================================
# building and finalizing a spec
# ====================================================================================================================
def make_invoice(
    rng: random.Random,
    template: str,
    uid: str,
    registry: NumberRegistry,
    *,
    n_items: int | None = None,
    max_desc: int = 140,
    omit: frozenset[str] = frozenset(),
    error: str | None = None,
    doc_type: DocumentType | None = None,
    force_currency: str | None = None,
    discount_shipping: bool | None = None,
    day: date | None = None,
    industry: str | None = None,
    long_text: bool = False,
) -> InvoiceSpec:
    """Create and finalize one document.

    ``error`` is one of ``eval.labels.ARITHMETIC_CODES``; ``omit`` names fields that are not printed at all;
    ``discount_shipping`` forces (True) or forbids (False) a discount and a shipping line, ``None`` draws them.
    """
    profile = PROFILES[template]
    market = MARKETS[rng.choice(profile.markets)]
    currency = force_currency or _weighted(rng, profile.currencies)
    receipt = template == "T5"
    if doc_type is None:
        doc_type = DocumentType.RECEIPT if receipt else DocumentType.INVOICE
    industry = industry or rng.choice(profile.industries)
    words = STORES[industry][0] if receipt else INDUSTRIES[industry][0]
    catalogue = STORE_ITEMS[industry] if receipt else INDUSTRY_ITEMS[industry]

    supplier = make_party(rng, market, words, long=rng.random() < 0.07, with_bank=not receipt)
    customer: Party | None = None
    if not receipt:
        cust_market = market if rng.random() < 0.85 else MARKETS[rng.choice(("us", "uk", "de", "au", "ca"))]
        person = rng.random() < (0.18 if template in {"T1", "T3", "T6"} else 0.04)
        customer = make_party(
            rng,
            cust_market,
            rng.choice(list(INDUSTRIES.values()))[0],
            person=person,
            long=rng.random() < 0.05,
            with_ids=rng.random() < 0.5,
        )
    elif rng.random() < 0.12:
        customer = Party(f"{rng.choice(market.first_names)} {rng.choice(market.surnames)}", [])
    ship_to = None
    if template == "T1" and customer and customer.name and rng.random() < 0.3:
        ship_to = make_party(rng, market, ("Warehouse", "Receiving", "Distribution"), with_ids=False)
        suffix = rng.choice(("Warehouse", "Receiving Dept.", "Site 2"))
        ship_to.name = customer.name if rng.random() < 0.5 else f"{customer.name} - {suffix}"
    if "customer_name" in omit:
        customer, ship_to = None, None

    invoice_day = day or random_date(rng)
    number = make_number(
        rng, template, invoice_day, supplier.name or "", registry, credit_note=doc_type is DocumentType.CREDIT_NOTE
    )
    terms = rng.choice((0, 7, 14, 15, 30, 30, 30, 45, 60)) or 14
    due = None
    if not receipt and rng.random() < profile.due_probability:
        due = invoice_day + timedelta(days=terms)
    service_date = None
    if template in {"T2", "T3", "T4"} and rng.random() < 0.55:
        service_date = invoice_day - timedelta(days=rng.randint(1, 20))

    items = make_items(
        rng,
        catalogue,
        _item_count(rng, n_items),
        profile.price_scale,
        max_desc=max_desc,
        receipt=receipt,
        template=template,
        long_text=long_text,
    )
    if receipt:
        for it in items:
            it.price_printed = it.quantity != 1 and rng.random() < 0.85

    scheme, rates, combined = _tax_setup(rng, template, market, currency)
    if scheme == "classes":
        for it in items:
            it.vat_class = 0 if rng.random() < 0.7 else 1
    if discount_shipping is None:
        has_discount = rng.random() < profile.discount_probability
        has_shipping = rng.random() < profile.shipping_probability
    else:
        has_discount = has_shipping = discount_shipping
    discount_pct = discount_fixed = shipping = None
    if has_discount and scheme != "classes":
        if rng.random() < 0.65:
            discount_pct = Decimal(rng.choice(("5", "10", "10", "15", "20", "7.5")))
        else:
            discount_fixed = Decimal(rng.choice((10, 15, 20, 25, 30, 50, 75, 100)))
    if has_shipping:
        shipping = q2(Decimal(str(rng.choice((5, 7.5, 9.95, 12, 15, 18.5, 25, 32, 45, 60)))))

    extra: dict[str, Any] = {
        "po": f"PO-{digits(rng, 5)}" if rng.random() < 0.45 else None,
        "customer_no": f"C{digits(rng, 5)}" if rng.random() < 0.5 else None,
        "terms_days": terms,
        "note": rng.choice(NOTES).format(days=terms, n4=digits(rng, 4), n5=digits(rng, 5)),
        "place_of_supply": _place_of_supply(customer),
        "cashier": rng.choice(market.first_names),
        "register": f"{rng.randint(1, 9):02d}",
        "card_last4": digits(rng, 4),
        "auth": digits(rng, 6),
        "barcode": digits(rng, 12),
        "store_no": digits(rng, 4),
        "clock": f"{rng.randint(8, 21):02d}:{rng.randint(0, 59):02d}",
        "industry": industry,
        "doc_ref": f"{digits(rng, 4)}-{digits(rng, 4)}",
    }
    spec = InvoiceSpec(
        uid=uid,
        template=template,
        doc_type=doc_type,
        market=market.key,
        currency=currency,
        currency_printed="currency" not in omit,
        invoice_number=None if "invoice_number" in omit else number,
        invoice_date=None if "invoice_date" in omit else invoice_day,
        due_date=due,
        service_date=service_date,
        supplier=supplier,
        customer=customer,
        ship_to=ship_to,
        items=items,
        tax_scheme="none" if "tax" in omit else scheme,
        tax_rates=[Decimal(r) for r in rates],
        discount_pct=discount_pct,
        discount_fixed=discount_fixed,
        shipping_amount=shipping,
        combined_tax_row=combined and "tax" not in omit,
        rate_printed=template in {"T2", "T4", "T5"} or rng.random() < 0.88,
        omit=omit,
        error=None,
        extra=extra,
    )
    if (
        "supplier_name" in omit
    ):  # a logo-only letterhead: the name is not printed, and neither is anything derived from it
        supplier.name = None
        supplier.email = supplier.web = None
    if error is not None:
        spec.error = plan_error(rng, spec, error)
    finalize(spec)
    return spec


def _place_of_supply(customer: Party | None) -> str:
    if customer is None or not customer.state_code:
        return "Maharashtra (27)"
    inverse = {v: k for k, v in GST_STATES.items()}
    return f"{inverse.get(customer.state_code, 'Maharashtra')} ({customer.state_code})"


def _tax_setup(rng: random.Random, template: str, market: Market, currency: str) -> tuple[str, tuple[str, ...], bool]:
    """(scheme, rates, combined-rate row printed) for a document of this template."""
    if template == "T1":
        roll = rng.random()
        if roll < 0.05:
            return "single", ("0",), False
        if roll < 0.28:
            return "pair", (rng.choice(("4", "5", "6", "6.25")), rng.choice(("1", "1.5", "2", "2.25"))), False
        return "single", (rng.choice(US_RATES),), False
    if template == "T2":
        rates = EU_RATES[market.key]
        if rng.random() < 0.15:
            return "classes", (rates[0], rates[1]), False
        if rng.random() < 0.05:
            return "single", ("0",), False
        return "single", (rates[0] if rng.random() < 0.85 else rates[1],), False
    if template == "T3":
        roll = rng.random()
        return "single", (("20",) if roll < 0.84 else ("5",) if roll < 0.94 else ("0",)), False
    if template == "T4":
        slab, half = rng.choice(GST_SLABS)
        if rng.random() < 0.2:
            return "igst", (slab,), False
        return "gst_split", (half, half), rng.random() < 0.5
    if template == "T5":
        scheme, rates = rng.choice(CA_TAX)
        return scheme, rates, False
    return "single", (("10",) if currency == "AUD" else ("0",)), False


# --------------------------------------------------------------------------------------------------------------------
# arithmetic and errors
# --------------------------------------------------------------------------------------------------------------------
SINGLE_TAX_KIND = {"T1": "tax", "T2": "vat", "T3": "vat", "T4": "igst", "T5": "gst", "T6": "gst"}


def plan_error(rng: random.Random, spec: InvoiceSpec, kind: str) -> ErrorSpec:
    """Decide how a document will be wrong (the numbers are produced by ``finalize``)."""
    n = len(spec.items)
    magnitude = Decimal(str(round(rng.uniform(1.0, 9.5), 2)))
    if kind == "TOTAL_MISMATCH":
        variant = rng.choice(("offset", "offset", "forgot_tax", "transposed", "forgot_shipping"))
        if variant == "forgot_shipping" and spec.shipping_amount is None:
            variant = "offset"
        if variant == "forgot_tax" and spec.tax_scheme == "none":
            variant = "offset"
        return ErrorSpec(kind, variant, magnitude * rng.choice((-1, 1)) * rng.choice((1, 1, 10)))
    if kind == "LINE_ITEMS_SUM_MISMATCH":
        variant = rng.choice(("drop_row", "wrong_subtotal")) if n >= 3 else "wrong_subtotal"
        return ErrorSpec(kind, variant, magnitude * rng.choice((-1, 1)) * rng.choice((2, 5, 9)), rng.randrange(n))
    if kind == "LINE_ITEM_MATH":
        priced = [i for i, it in enumerate(spec.items) if it.price_printed]
        if not priced:  # a receipt line without a printed unit price cannot show a multiplication error
            spec.items[0].price_printed = True
            priced = [0]
        return ErrorSpec(
            kind, "wrong_amount", magnitude * rng.choice((-1, 1)) * rng.choice((1, 3, 6)), rng.choice(priced)
        )
    raise ValueError(f"unknown error kind {kind}")


def _transpose(total: Decimal) -> Decimal | None:
    """The total with two adjacent integer digits swapped (a classic keying error), if that changes it enough."""
    chars = list(f"{total:.2f}")
    for i in range(len(chars) - 5, -1, -1):
        if chars[i].isdigit() and chars[i + 1].isdigit() and chars[i] != chars[i + 1]:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            swapped = Decimal("".join(chars))
            if abs(swapped - total) >= Decimal("0.5"):
                return swapped
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return None


def _tax_lines(spec: InvoiceSpec, base: Decimal) -> list[TaxLine]:
    scheme, rates = spec.tax_scheme, spec.tax_rates
    if scheme == "single":
        hst = spec.template == "T5" and rates[0] in {Decimal(13), Decimal(15)}
        kind = "hst" if hst else SINGLE_TAX_KIND[spec.template]
        return [TaxLine(kind, rates[0], q2(base * rates[0] / HUNDRED), base)]
    if scheme == "igst":
        return [TaxLine("igst", rates[0], q2(base * rates[0] / HUNDRED), base)]
    if scheme == "pair":
        second = "qst" if rates[1] == Decimal("9.975") else "pst"
        kinds = ("state", "county") if spec.template == "T1" else ("gst", second)
        return [TaxLine(k, r, q2(base * r / HUNDRED), base) for k, r in zip(kinds, rates, strict=True)]
    if scheme == "gst_split":
        return [TaxLine(k, r, q2(base * r / HUNDRED), base) for k, r in zip(("cgst", "sgst"), rates, strict=True)]
    if scheme == "classes":
        lines = []
        for cls, rate in enumerate(rates):
            class_base = sum((it.amount for it in spec.items if it.vat_class == cls), Decimal(0))
            if class_base > 0:
                lines.append(TaxLine("vat", rate, q2(class_base * rate / HUNDRED), class_base))
        return lines
    return []


def finalize(spec: InvoiceSpec) -> None:
    """Compute every printed money field from the inputs, then apply the intentional error (idempotent)."""
    err = spec.error
    items = spec.items
    for idx, it in enumerate(items):
        it.amount = q2(it.quantity * it.unit_price)
        if err and err.kind == "LINE_ITEM_MATH" and idx == err.index % len(items):
            candidate = q2(it.amount + err.delta)
            it.amount = candidate if candidate > 0 and candidate != it.amount else q2(it.amount + abs(err.delta) + 1)
    true_subtotal = sum((it.amount for it in items), Decimal(0))
    subtotal = true_subtotal
    spec.dropped = None
    if err and err.kind == "LINE_ITEMS_SUM_MISMATCH":
        if err.variant == "drop_row" and len(items) >= 3:
            spec.dropped = err.index % len(items)
        else:
            subtotal = q2(max(true_subtotal + err.delta, Decimal("1.00")))
            if abs(subtotal - true_subtotal) <= Decimal("0.5"):
                subtotal = true_subtotal + Decimal("7.35")
    discount = Decimal(0)
    if spec.discount_pct is not None:
        discount = q2(subtotal * spec.discount_pct / HUNDRED)
    elif spec.discount_fixed is not None:
        discount = min(spec.discount_fixed, subtotal)
    base = subtotal - discount
    taxes = _tax_lines(spec, base)
    shipping = spec.shipping_amount or Decimal(0)
    total = base + shipping + sum((t.amount for t in taxes), Decimal(0))
    if err and err.kind == "TOTAL_MISMATCH":
        total = _mismatched_total(err, base, shipping, taxes, total)

    spec.subtotal = None if "subtotal" in spec.omit else q2(subtotal)
    spec.discount = q2(discount) if discount > 0 else None
    spec.shipping = q2(shipping) if spec.shipping_amount is not None else None
    spec.taxes = taxes
    spec.total = None if "total" in spec.omit else q2(total)


def _mismatched_total(
    err: ErrorSpec, base: Decimal, shipping: Decimal, taxes: list[TaxLine], total: Decimal
) -> Decimal:
    """The wrong grand total that gets printed (always at least 0.50 away from the right one)."""
    tax_sum = sum((t.amount for t in taxes), Decimal(0))
    wrong = total
    if err.variant == "forgot_tax" and tax_sum >= Decimal("1.00"):
        wrong = base + shipping
    elif err.variant == "forgot_shipping" and shipping >= Decimal("1.00"):
        wrong = total - shipping
    elif err.variant == "transposed":
        wrong = _transpose(total) or total
    if abs(wrong - total) < Decimal("0.5"):
        wrong = total + err.delta
    if abs(wrong - total) < Decimal("0.5"):
        wrong = total + Decimal("12.40")
    return q2(wrong)
