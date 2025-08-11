using UnityEngine;

[CreateAssetMenu(fileName = "QuayGradeMaterialMap", menuName = "Quay Visualization/Grade Material Map")]
public class GradeMaterialMap : ScriptableObject
{
    public Material gradeA;
    public Material gradeB;
    public Material gradeC;
    public Material gradeD;
    public Material gradeE;
    public Material gradeN;
    public Material engagedMaterial;  // New green material for engaged quay walls

    public Material GetMaterial(QuayScoreGrade grade, bool isEngaged = false)
    {
        if (isEngaged)
            return engagedMaterial;

        return grade switch
        {
            QuayScoreGrade.A => gradeA,
            QuayScoreGrade.B => gradeB,
            QuayScoreGrade.C => gradeC,
            QuayScoreGrade.D => gradeD,
            QuayScoreGrade.E => gradeE,
            QuayScoreGrade.N => gradeN,
            _ => gradeN
        };
    }
}