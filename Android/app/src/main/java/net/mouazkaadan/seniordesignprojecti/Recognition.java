package net.mouazkaadan.seniordesignprojecti;

class Recognition {

    private String title;
    private Float confidence;

    public Recognition(String title, Float confidence) {
        this.title = title;
        this.confidence = confidence;
    }


    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }
}