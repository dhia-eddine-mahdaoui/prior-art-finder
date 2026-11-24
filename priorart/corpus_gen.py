"""Deterministic generator for the synthetic patent-style corpus.

Everything here is invented: assignees are fictional, document ids are
synthetic (SYN-xxxx), and text is assembled from templates. Nothing is copied
from real patents. The generator also writes the sample disclosures and the
relevance labels (qrels), which are known by construction.

Run:  python -m priorart gen
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from .corpus_extra import EXTRA, disclosure_text

DATA = Path(__file__).resolve().parent.parent / "data"

AREAS = {
    "battery": {
        "subjects": ["battery pack", "battery module", "energy storage assembly", "thermal management system"],
        "cpc": ["H01M 10/613", "H01M 10/625", "H01M 10/6556", "H01M 50/20", "B60L 58/26", "H01M 10/655"],
        "assignees": ["Halvorsen-Brek Energy", "Lumenfall Cells Inc.", "Northgate Power Labs", "Oryx Valley Storage", "Tamsin Dynamics"],
        "benefits": ["reduces temperature spread between cells", "extends cycle life", "improves fast-charging performance", "lowers parasitic cooling power"],
        "general": [
            ("air-cooled housing", "a module housing with forced-air cooling ducts between cell rows"),
            ("heat spreader foils", "an aluminium heat-spreading foil wrapped around each cylindrical cell"),
            ("thermal interface pad", "a compressible thermal interface pad between the cell and a cooling surface"),
            ("immersion cooling", "a dielectric fluid in which the cells are partially immersed"),
            ("cell balancing", "a balancing circuit that equalises the state of charge of series-connected cells"),
            ("thermistor network", "a plurality of thermistors bonded to cell surfaces"),
            ("busbar heat sink", "a busbar with an integrated heat sink"),
            ("sealed enclosure", "a sealed pack enclosure with a pressure-equalising membrane"),
            ("derated charging", "a charging profile that derates current as cell temperature increases"),
            ("heating film", "a resistive heating film attached to the pack underside"),
            ("graphite sheet", "a pyrolytic graphite sheet conducting heat toward a side wall"),
            ("fan control", "a fan whose duty cycle follows the highest measured cell temperature"),
            ("heat pipe", "a heat pipe transferring heat from the module core to a condenser section"),
            ("aerogel cover", "an aerogel insulation blanket covering the pack lid"),
        ],
        "special": {
            "B1": ["a phase-change material layer disposed between adjacent battery cells to absorb heat",
                   "a paraffin composite interposed between neighbouring cells, the composite storing latent heat during melting",
                   "a latent-heat storage sheet sandwiched between adjoining cells of the module"],
            "B2": ["a cooling plate having serpentine microchannels through which coolant circulates",
                   "a cold plate with meandering micro-channels carrying liquid coolant beneath the cells",
                   "a liquid-cooled base plate defining a winding fine-channel flow path"],
            "B3": ["a controller that adjusts coolant flow rate using a predictive model of the cell temperature gradient",
                   "a control unit setting pump speed from a forecast of temperature differences across the cells",
                   "a processor regulating coolant throughput according to a model predicting thermal gradients between cells"],
            "B4": ["detecting thermal runaway when a rate of temperature rise exceeds a threshold",
                   "a monitoring circuit flagging a cell failure event upon a temperature slope above a limit",
                   "identifying an overheating cascade from the time derivative of measured cell temperature"],
            "B5": ["an intumescent barrier and a venting channel that directs ejected gas away from adjacent cells",
                   "an expanding fire-retardant layer together with an exhaust duct guiding vented gases out of the housing",
                   "a flame-blocking swellable partition and a gas outlet path leading emissions away from neighbours"],
            "B6": ["a reversible heat pump that pre-heats the battery pack in cold ambient conditions",
                   "a bidirectional refrigerant circuit warming the pack before charging at low temperature",
                   "a heat-pumping loop operable in heating mode to raise pack temperature in cold weather"],
        },
        "decoys": ["a cooling plate having straight macro-channels fed by a fixed-speed pump",
                   "a phase-change material tank located outside the pack for domestic hot water",
                   "a thermal runaway alarm triggered only by a gas concentration sensor"],
    },
    "sensors": {
        "subjects": ["sensor node", "condition monitoring device", "wireless monitoring system", "machine health sensor"],
        "cpc": ["G01H 1/00", "G01P 15/09", "G01M 7/02", "H04Q 9/00", "G05B 23/02", "H02N 2/18"],
        "assignees": ["Pellucid Sensing GmbH", "Quillon Instruments", "Rotari Measurement Co.", "Strandwick Controls", "Vesper Metrology"],
        "benefits": ["reduces maintenance visits", "extends battery life", "shortens fault detection latency", "lowers installation cost"],
        "general": [
            ("ultrasonic flow meter", "an ultrasonic transducer pair measuring transit time through the pipe wall"),
            ("pressure transmitter", "a diaphragm pressure transmitter with a current loop output"),
            ("thermocouple assembly", "a mineral-insulated thermocouple with a thermowell"),
            ("fibre sensing", "a fibre Bragg grating strain sensor bonded to the structure"),
            ("gas detection", "an electrochemical gas detection cell with a replaceable filter"),
            ("level radar", "a guided-wave radar probe for tank level measurement"),
            ("calibration memory", "a calibration routine storing correction coefficients in non-volatile memory"),
            ("long-range gateway", "a gateway aggregating readings from field devices over a long-range radio link"),
            ("safety barrier", "an intrinsically safe barrier limiting energy delivered to a hazardous area"),
            ("scheduled logger", "a battery-powered data logger with scheduled wake-up"),
            ("proximity sensing", "an inductive proximity sensor detecting a rotating target"),
            ("self-diagnostics", "a self-diagnostic function reporting sensor element failure"),
            ("humidity compensation", "a humidity compensation step applied to capacitive readings"),
            ("edge unit", "an edge computing unit forwarding alarms to a supervisory system"),
        ],
        "special": {
            "S1": ["a piezoelectric energy harvester that powers the sensor node from machine vibration",
                   "a vibration-driven piezo generator supplying electrical power to the wireless node",
                   "an energy-scavenging transducer converting mechanical oscillation into supply power for the device"],
            "S2": ["a MEMS accelerometer sampled at an adaptive rate that increases when vibration amplitude rises",
                   "a micromachined acceleration transducer whose sampling frequency is raised upon larger oscillation levels",
                   "an inertial chip read at a variable acquisition rate scaled to vibration intensity"],
            "S3": ["on-device spectral feature extraction using a fast Fourier transform and an autoencoder computing an anomaly score",
                   "an embedded processor deriving frequency-domain features and a neural reconstruction-error measure of abnormality",
                   "local spectrum analysis followed by a learned model scoring deviation from normal behaviour at the node"],
            "S4": ["a low-power mesh radio that transmits only when the anomaly score exceeds a threshold",
                   "a duty-cycled wireless mesh link sending data solely upon an abnormality indication",
                   "a sleeping network transceiver woken to report only when a fault indicator is raised"],
            "S5": ["temperature compensation of accelerometer drift using a reference thermistor",
                   "correcting sensor offset drift with readings from a nearby thermal resistor",
                   "a thermistor-based correction applied to remove bias variation of the acceleration sensor"],
            "S6": ["time-synchronised timestamps shared across nodes of the mesh",
                   "a common clock reference aligning sample times between distributed nodes",
                   "clock alignment so measurements from several devices carry coherent time labels"],
        },
        "decoys": ["a wired vibration sensor powered from a mains supply that transmits continuously",
                   "a piezoelectric accelerometer for laboratory modal testing with a fixed sampling rate",
                   "a cloud-based fault analysis service receiving raw waveforms"],
    },
    "coatings": {
        "subjects": ["coating composition", "protective coating system", "coated article", "corrosion protection scheme"],
        "cpc": ["C09D 5/08", "C09D 163/00", "C23F 11/08", "C09D 7/61", "B05D 3/06", "C08K 3/04"],
        "assignees": ["Ashbourne Coatings Ltd.", "Kestrel Surface Works", "Maren Polymer Group", "Ostrava Finish Labs", "Pinecrest Materials"],
        "benefits": ["extends service life in marine environments", "reduces maintenance repainting", "lowers solvent emissions", "improves adhesion on steel"],
        "general": [
            ("thermal barrier", "a yttria-stabilised zirconia layer deposited by plasma spraying"),
            ("hydrophobic topcoat", "a fluorinated topcoat repelling water"),
            ("powder coating", "a thermosetting polyester powder applied electrostatically"),
            ("sol-gel layer", "a sol-gel silica layer applied by dip coating"),
            ("anti-fouling paint", "a copper-free anti-fouling paint releasing a biocide slowly"),
            ("intumescent paint", "an intumescent paint layer that swells when exposed to heat"),
            ("chromium-free pretreatment", "a chromium-free conversion pretreatment of an aluminium substrate"),
            ("carbon film", "a diamond-like carbon film deposited by physical vapour deposition"),
            ("nickel plating", "an electroless nickel-phosphorus plating"),
            ("thickness gauge", "a non-contact eddy current gauge for measuring film thickness"),
            ("abrasion resistance", "ceramic particles dispersed in a polyurethane binder for abrasion resistance"),
            ("anodising", "a sealed anodic oxide layer on an aluminium alloy"),
            ("low-solvent binder", "a water-borne acrylic dispersion with reduced solvent content"),
            ("shielding paint", "a silver-filled conductive coating for electromagnetic shielding"),
        ],
        "special": {
            "C1": ["microcapsules containing a corrosion inhibitor that release when a crack forms in the coating",
                   "encapsulated inhibitor-filled shells that rupture upon cracking and discharge their payload",
                   "tiny capsules holding an anti-rust agent liberated on fracture of the film"],
            "C2": ["graphene oxide platelets aligned parallel to the surface within an epoxy matrix",
                   "flake-like graphitic nanosheets oriented in-plane inside a cured epoxy resin",
                   "layered carbon nanoplatelets lying flat in a thermoset binder forming a barrier"],
            "C3": ["a zinc-rich primer layer providing sacrificial cathodic protection to the steel substrate",
                   "an undercoat loaded with zinc dust protecting the steel galvanically",
                   "a metallic-zinc filled base layer acting as a sacrificial anode for the pipe wall"],
            "C4": ["curing by ultraviolet light-emitting diodes to a dry film thickness of 80 to 120 micrometres",
                   "UV-LED irradiation hardening the film to a thickness between 80 and 120 micrometres",
                   "photo-curing with solid-state UV lamps yielding a dry layer of about one hundred micrometres"],
            "C5": ["an embedded electrochemical impedance sensor that monitors degradation of the coating",
                   "an in-film impedance spectroscopy probe tracking deterioration of the protective layer",
                   "a built-in electrode pair measuring AC impedance to follow coating ageing"],
            "C6": ["a silane adhesion promoter applied to a grit-blasted surface",
                   "an organosilane coupling agent on an abrasively blasted metal surface",
                   "a silane primer treatment over sand-blasted steel to improve bonding"],
        },
        "decoys": ["a self-healing polymer containing reversible hydrogen bonds without any capsules",
                   "a graphene-filled conductive ink for printed electronics",
                   "a zinc electroplating bath for fasteners"],
    },
    "additive": {
        "subjects": ["additive manufacturing apparatus", "powder bed fusion system", "laser melting machine", "layer-wise fabrication method"],
        "cpc": ["B22F 10/28", "B22F 10/85", "B33Y 10/00", "B33Y 50/02", "B29C 64/153", "B23K 26/342"],
        "assignees": ["Tessara Fabrication", "Ironleaf Additive", "Delmore Laser Systems", "Brightwater Machines", "Quarry Hill Technologies"],
        "benefits": ["reduces porosity in finished parts", "improves build yield", "shortens qualification time", "lowers scrap rate"],
        "general": [
            ("support structures", "lattice support structures designed for easy removal"),
            ("powder recycling", "a sieving station returning unfused powder to the hopper"),
            ("scan strategy", "a stripe scan strategy rotated by sixty-seven degrees between layers"),
            ("stress relief", "a post-build stress-relief heat treatment"),
            ("pore closing", "hot isostatic pressing to close internal pores"),
            ("multi-laser zoning", "a plurality of lasers each assigned to a zone of the build plate"),
            ("graded lattice", "a topology-optimised lattice with graded cell size"),
            ("wire deposition", "a wire-fed arc deposition head mounted on a robot arm"),
            ("binder jetting", "a binder jetting printhead depositing a binder on a powder layer"),
            ("surface finishing", "an abrasive flow process polishing internal channels"),
            ("orientation planning", "an orientation planner minimising overhang area"),
            ("roller spreading", "a counter-rotating roller spreading powder"),
            ("distortion simulation", "a thermal distortion simulation used to pre-compensate part geometry"),
            ("part marking", "a data matrix mark printed on each part for traceability"),
        ],
        "special": {
            "A1": ["a coaxial photodiode that measures melt pool emission through the laser optics",
                   "an on-axis optical detector sensing radiation from the molten pool via the scanning optics",
                   "a light sensor sharing the laser beam path to record melt-zone brightness"],
            "A2": ["a high-speed camera whose thermal images are used to determine melt pool width",
                   "a fast infrared imager from which the molten track width is computed",
                   "rapid thermographic imaging yielding the dimension of the molten zone"],
            "A3": ["a controller that adjusts laser power for each scan vector in real time based on the monitored signal",
                   "closed-loop regulation of beam power per scan segment according to sensed melt data",
                   "a feedback unit modulating laser output on every hatch line during the build"],
            "A4": ["a line sensor mounted on the recoater that scans each powder layer to detect streaks",
                   "a linear optical scanner carried by the spreader inspecting the powder bed for defects",
                   "a recoating-blade mounted array detector checking layer uniformity and grooves"],
            "A5": ["an inert gas flow with an oxygen sensor and a spatter shield above the build plate",
                   "a shielding gas stream monitored by an oxygen analyser and a spatter deflector",
                   "argon flow across the bed, an oxygen monitor, and a baffle diverting ejected particles"],
            "A6": ["preheating the build plate with a temperature gradient across its area",
                   "a heated platform held at a non-uniform temperature profile",
                   "a substrate heater producing a graded thermal profile"],
        },
        "decoys": ["a powder bed fusion machine with fixed laser power and post-build inspection by tomography",
                   "a coaxial camera used only for recording build documentation",
                   "a recoater with a brush made of carbon fibre"],
    },
}

DISCLOSURES = [
    {
        "id": "D1", "area": "battery", "title": "Predictive pack thermal management with latent-heat layers",
        "priority_date": "2023-03-10", "classes": ["H01M"],
        "text": (
            "We propose a thermal management arrangement for a traction battery pack used in electric vehicles. "
            "A phase-change material layer is disposed between adjacent battery cells to absorb heat during fast charging. "
            "Underneath the cells, a cooling plate having serpentine microchannels circulates coolant. "
            "A controller adjusts coolant flow rate using a predictive model of the cell temperature gradient, rather than reacting to absolute temperature. "
            "Thermal runaway is detected when a rate of temperature rise exceeds a threshold, in which case an intumescent barrier and a venting channel direct ejected gas away from adjacent cells. "
            "In cold ambient conditions a reversible heat pump pre-heats the battery pack before charging."
        ),
    },
    {
        "id": "D2", "area": "sensors", "title": "Self-powered vibration node with on-device anomaly detection",
        "priority_date": "2022-11-04", "classes": ["G01H", "G01P", "G05B"],
        "text": (
            "The disclosure describes a wireless condition monitoring node for rotating machinery in a plant. "
            "A piezoelectric energy harvester powers the sensor node from machine vibration, so no batteries are replaced. "
            "A MEMS accelerometer is sampled at an adaptive rate that increases when vibration amplitude rises. "
            "On-device spectral feature extraction using a fast Fourier transform feeds an autoencoder computing an anomaly score. "
            "A low-power mesh radio transmits only when the anomaly score exceeds a threshold; otherwise the node sleeps. "
            "Temperature compensation of accelerometer drift uses a reference thermistor, and time-synchronised timestamps are shared across nodes of the mesh."
        ),
    },
    {
        "id": "D3", "area": "coatings", "title": "Self-healing barrier coating with in-film monitoring for pipelines",
        "priority_date": "2023-06-21", "classes": ["C09D", "C23F"],
        "text": (
            "We present a multilayer protective coating for buried steel pipelines. "
            "Microcapsules containing a corrosion inhibitor release when a crack forms in the coating. "
            "Graphene oxide platelets aligned parallel to the surface within an epoxy matrix lengthen the diffusion path of water and chloride. "
            "A zinc-rich primer layer provides sacrificial cathodic protection to the steel substrate, applied over a silane adhesion promoter on a grit-blasted surface. "
            "The topcoat is cured by ultraviolet light-emitting diodes to a dry film thickness of 80 to 120 micrometres. "
            "An embedded electrochemical impedance sensor monitors degradation of the coating during service."
        ),
    },
    {
        "id": "D4", "area": "additive", "title": "Closed-loop melt pool control for powder bed fusion",
        "priority_date": "2023-01-30", "classes": ["B22F", "B33Y"],
        "text": (
            "This application relates to a laser powder bed fusion machine with in-situ quality control. "
            "A coaxial photodiode measures melt pool emission through the laser optics. "
            "A high-speed camera captures thermal images that are used to determine melt pool width. "
            "A controller adjusts laser power for each scan vector in real time based on the monitored signal. "
            "Before exposure, a line sensor mounted on the recoater scans each powder layer to detect streaks. "
            "An inert gas flow with an oxygen sensor and a spatter shield is arranged above the build plate, and the build plate is preheated with a temperature gradient across its area."
        ),
    },
]

SYNONYM_EDITS = [("disclosed", "described"), ("In some embodiments", "In certain embodiments"),
                 ("improves", "enhances"), ("reduces", "lowers"), ("includes", "has")]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _make_doc(rng, area, feats, shorts, filing, assignee=None, key=None, present=None):
    a = AREAS[area]
    subject = rng.choice(a["subjects"])
    feats = list(feats)
    title = f"{_cap(subject)} with {shorts[0]}" + (f" and {shorts[1]}" if len(shorts) > 1 and rng.random() < 0.5 else "")
    benefit = rng.choice(a["benefits"])
    t = rng.randrange(3)
    if t == 0:
        abstract = (f"A {subject} is disclosed that includes {feats[0]}. In some embodiments the {subject} further includes "
                    f"{feats[1]}" + (f" and {feats[2]}" if len(feats) > 2 else "") + f". The arrangement {benefit}.")
    elif t == 1:
        abstract = (f"The present disclosure relates to a {subject}. The {subject} comprises {feats[0]} and {feats[1]}. "
                    + (f"Additionally, {feats[2]} is provided. " if len(feats) > 2 else "") + f"This {benefit}.")
    else:
        abstract = (f"Embodiments of a {subject} are described, characterised by {feats[0]}. "
                    + (f"Variants include {feats[2]}. " if len(feats) > 2 else "") + f"The design {benefit} while keeping cost low.")
    claims = [f"1. A {subject} comprising: " + "; ".join(feats[:2]) + (f"; and {feats[2]}." if len(feats) > 2 else ".")]
    rest = feats[3:]
    for i, f in enumerate(rest[:3]):
        claims.append(f"{i + 2}. The {subject} of claim {i + 1 if i else 1}, further comprising {f}.")
    cpc = [rng.choice(a["cpc"])]
    for c in rng.sample(a["cpc"], 2):
        if c not in cpc and rng.random() < 0.6:
            cpc.append(c)
    return {"id": None, "title": title, "abstract": abstract, "claims": claims, "cpc": cpc,
            "filing_date": filing.isoformat(), "assignee": assignee or rng.choice(a["assignees"]),
            "area": area, "_key": key, "_present": present or {}}


def _near_dup(rng, d, key):
    n = json.loads(json.dumps(d))
    n["_key"] = key
    n["filing_date"] = (date.fromisoformat(d["filing_date"]) + timedelta(days=rng.randrange(20, 180))).isoformat()
    if rng.random() < 0.5:
        n["assignee"] = rng.choice(AREAS[d["area"]]["assignees"])
    for old, new in SYNONYM_EDITS:
        if old in n["abstract"] and rng.random() < 0.7:
            n["abstract"] = n["abstract"].replace(old, new, 1)
    if len(n["claims"]) > 2:
        dropped = 3 + len(n["claims"]) - 2   # index of the feature carried only by the last dependent claim
        n["claims"] = n["claims"][:-1]
        n["_present"] = {k: i for k, i in n["_present"].items() if i != dropped}
    n["title"] = n["title"] + " (continuation)" if rng.random() < 0.5 else n["title"]
    return n


def _rand_date(rng, lo=(2009, 1, 1), hi=(2022, 12, 31)):
    a, b = date(*lo), date(*hi)
    return a + timedelta(days=rng.randrange((b - a).days))


def _all_disclosures():
    """Original D1-D4 (features live in AREAS) followed by the extra disclosures D5-D16."""
    out = []
    for d in DISCLOSURES:
        a = AREAS[d["area"]]
        out.append({**d, "split": "dev", "spec": {k: list(v) for k, v in a["special"].items()}, "decoys": a["decoys"]})
    for e in EXTRA:
        out.append({"id": e["id"], "area": e["area"], "title": e["title"], "priority_date": e["priority"],
                    "classes": e["classes"], "text": disclosure_text(e), "split": e["split"],
                    "spec": {f"{e['id']}.{i + 1}": list(f) for i, f in enumerate(e["feats"])}, "decoys": e["decoys"]})
    return out


def build_corpus(seed: int = 7):
    rng = random.Random(seed)
    docs, labels, gold_spec = [], {}, {}
    for disc in _all_disclosures():
        area = disc["area"]
        a = AREAS[area]
        did = disc["id"]
        spec = disc["spec"]
        keys = list(spec)
        gen = a["general"]
        grades = {}
        gold_spec[did] = {k: v[0] for k, v in spec.items()}

        def gen_pick(n):
            g = rng.sample(gen, n)
            return [x[1] for x in g], [x[0] for x in g]

        def present_of(ks):
            return {k: i for i, k in enumerate(ks)}

        # close prior art: five of six features each, increasingly paraphrased
        close = []
        for ci, (omit, vmix) in enumerate([(keys[5], [1, 1, 1, 1, 1]), (keys[2], [0, 1, 1, 0, 1]), (keys[4], [2, 2, 1, 2, 2])]):
            ks = [k for k in keys if k != omit]
            rng.shuffle(ks)
            gf, gs = gen_pick(1)
            feats = [spec[k][vmix[i]] for i, k in enumerate(ks)] + gf
            d = _make_doc(rng, area, feats, gs, _rand_date(rng, (2016, 1, 1), (2022, 1, 31)), key=f"{did}-close{ci}", present=present_of(ks))
            docs.append(d); close.append(d); grades[d["_key"]] = 3
        for ci in (0, 1):
            d = _near_dup(rng, close[ci], f"{did}-dup{ci}")
            docs.append(d); grades[d["_key"]] = 3
        # partial prior art: 2-3 features among general ones
        for pi, n in enumerate((3, 3, 2)):
            ks = rng.sample(keys, n)
            gf, gs = gen_pick(3)
            feats = [spec[k][rng.choice([1, 2])] for k in ks] + gf
            order = list(range(len(feats)))
            rng.shuffle(order)
            feats = [feats[j] for j in order]
            present = {ks[j]: order.index(j) for j in range(n)}
            d = _make_doc(rng, area, feats, gs, _rand_date(rng, (2012, 1, 1), (2022, 1, 31)), key=f"{did}-part{pi}", present=present)
            docs.append(d); grades[d["_key"]] = 2 if n == 3 else 1
        # hard decoys: vocabulary overlap, different mechanism
        for di, dec in enumerate(disc["decoys"]):
            gf, gs = gen_pick(3)
            d = _make_doc(rng, area, [dec] + gf, gs, _rand_date(rng, (2012, 1, 1), (2022, 6, 30)), key=f"{did}-decoy{di}")
            docs.append(d); grades[d["_key"]] = 0
        # strong matches filed after the priority date (must be excluded by the date filter)
        for li in range(2):
            ks = rng.sample(keys, 5)
            gf, gs = gen_pick(1)
            feats = [spec[k][rng.choice([0, 1])] for k in ks] + gf
            d = _make_doc(rng, area, feats, gs, _rand_date(rng, (2023, 8, 1), (2024, 6, 30)), key=f"{did}-late{li}", present=present_of(ks))
            docs.append(d); grades[d["_key"]] = 0
        labels[did] = grades

    # background documents and background near-duplicates
    for area, a in AREAS.items():
        bg = []
        for i in range(60):
            n = rng.choice([3, 4, 4, 5])
            g = rng.sample(a["general"], n)
            d = _make_doc(rng, area, [x[1] for x in g], [x[0] for x in g], _rand_date(rng), key=f"bg-{area}-{i}")
            docs.append(d); bg.append(d)
        for i, src in enumerate(rng.sample(bg, 3)):
            docs.append(_near_dup(rng, src, f"bgdup-{area}-{i}"))

    rng.shuffle(docs)
    key2id, present_by_id = {}, {}
    for i, d in enumerate(docs, 1):
        d["id"] = f"SYN-{i:04d}"
        key2id[d.pop("_key")] = d["id"]
        present_by_id[d["id"]] = d.pop("_present")
    qrels = {did: {key2id[k]: g for k, g in gr.items()} for did, gr in labels.items()}
    # feature-level gold: which key feature of each disclosure is really disclosed in each document
    gold = {did: {"features": gold_spec[did],
                  "docs": {i: sorted(k for k in pr if k in gold_spec[did]) for i, pr in present_by_id.items()}}
            for did in gold_spec}
    return docs, qrels, gold


def write_all(seed: int = 7):
    DATA.mkdir(exist_ok=True)
    docs, qrels, gold = build_corpus(seed)
    with open(DATA / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    discl = [{k: v for k, v in d.items() if k in ("id", "title", "priority_date", "classes", "text", "split")}
             for d in _all_disclosures()]
    (DATA / "disclosures.json").write_text(json.dumps(discl, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (DATA / "qrels.json").write_text(json.dumps(qrels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # keep only positive presence per document to keep the file small
    slim = {q: {"features": g["features"], "docs": {i: k for i, k in g["docs"].items() if k}} for q, g in gold.items()}
    (DATA / "feature_gold.json").write_text(json.dumps(slim, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return len(docs)


if __name__ == "__main__":
    print(write_all(), "documents written")
