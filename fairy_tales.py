from enum import Enum

class FairyTale(Enum):
    Biberce = "biberce"
    Azdaja_i_Carev_Sin = "azdaja-i-carev-sin"
    Cardak_ni_na_Nebu_ni_na_Zemlji = "cardak-ni-na-nebu-ni-na-zemlji"
    Cela = "cela"
    Djavo_i_njegov_Segrt = "djavo-i-njegov-segrt"
    Devojka_brza_od_konja = "devojka-brza-od-konja"
    Devojka_Cara_nadmudrila = "devojka-cara-nadmudrila"
    Ero_s_onoga_Svijeta = "ero-s-onoga-svijeta"
    Tri_Prstena = "tri-prstena"
    Tri_Jegulje = "tri-jegulje"
    Zlatna_Jabuka_i_devet_Paunica = "zlatna-jabuka-i-devet-paunica"
    Zlatoruni_ovan = "zlatoruni-ovan"
    
    def get_url(self):
        base_url = "https://www.bajke.rs/"
        return f"{base_url}{self.value}/"
